# SPDX-License-Identifier: Apache-2.0
"""
MLX Embedding Model wrapper.

This module loads XLM-RoBERTa, BERT, and Qwen2-decoder embedding models with
native omlx implementations and every other embedding model through mlx-vlm.
"""

import gc
import inspect
import io
import json
import logging
from copy import deepcopy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_vlm.embedding_loader import load_embedding_model
from mlx_vlm.models.lfm2_colbert import Model as Lfm2Colbert
from mlx_vlm.models.llama_bidirec import Model as LlamaBidirec
from mlx_vlm.models.llama_nemotron_vl_embedding import Model as LlamaNemotron
from mlx_vlm.models.pooling import EmbeddingOutput as VlmEmbeddingOutput
from mlx_vlm.models.pooling import cls_pooling
from mlx_vlm.models.pooling import normalize_embeddings as normalize_token_embeddings
from mlx_vlm.models.pooling import pool_by_config
from mlx_vlm.models.qwen3_vl_embedding import Model as Qwen3VLEmbedding
from mlx_vlm.utils import get_model_path, load_audio, load_config, load_processor

from ..patches.modernbert_attention import patch_modernbert_attention
from ..utils.image import _decode_input_audio_data, load_image, validate_image_data_uri
from ..utils.model_loading import (
    ensure_model_code_trusted,
    normalize_encoder_quantization_config,
)
from ._siglip_embedding import (
    SiglipImageEncoder,
    load_siglip_vision,
    set_siglip_activation,
)
from .base_model import (
    ENCODER_BATCH_TOKEN_BUDGET,
    last_token_pool,
    mean_pooling,
    normalize_embeddings,
    token_budget_batches,
)

logger = logging.getLogger(__name__)

_DEFAULT_EMBEDDING_MAX_LENGTH = 512
_TOKENIZER_MAX_LENGTH_SENTINEL = 10**18
_CONTEXT_LENGTH_ATTRS = (
    "max_position_embeddings",
    "max_seq_len",
    "max_seq_length",
    "seq_length",
    "n_positions",
)


@dataclass(frozen=True)
class _ChatEmbeddingFamily:
    """How an mlx-vlm multimodal embedding family formats its inputs."""

    context_length: int
    """Context window from the model card. config.json only declares RoPE."""

    system_instruction: str | None = None
    add_generation_prompt: bool = False
    media_before_text: bool = False
    template_text: bool = False
    """Format text-only inputs with the chat template too."""


_CHAT_EMBEDDING_FAMILIES = {
    "embedding_gemma2": _ChatEmbeddingFamily(context_length=8192),
    "qwen3_vl_embedding": _ChatEmbeddingFamily(
        context_length=32768,
        system_instruction="Represent the user's input.",
        add_generation_prompt=True,
        media_before_text=True,
        template_text=True,
    ),
}

# The stock Qwen3-VL embedding checkpoints keep the qwen3_vl model_type.
_EMBEDDING_MODEL_TYPE_OVERRIDES = {
    "qwen3_vl": "qwen3_vl_embedding",
    "llama_nemotron_vl": "llama_nemotron_vl_embedding",
}

_POOLING_MODE_ALIASES = {
    "cls": "cls",
    "mean": "mean",
    "max": "max",
    "lasttoken": "lasttoken",
    "last_token": "lasttoken",
}
_POOLING_FLAG_MODES = {
    "pooling_mode_cls_token": "cls",
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_max_tokens": "max",
    "pooling_mode_lasttoken": "lasttoken",
}

# SigLIP's text tower pools the last position, so inputs pad to the full length.
_PAD_TO_MAX_LENGTH_MODEL_TYPES = frozenset({"siglip"})


def _colqwen_loader_config(config: dict[str, Any]) -> dict[str, Any]:
    from mlx_vlm.models.colqwen2_5 import TextConfig

    vlm = config["vlm_config"]
    text = vlm.get("text_config")
    text = text if isinstance(text, dict) else {}
    effective = deepcopy(config)
    effective.update(deepcopy(vlm))
    for field in fields(TextConfig):
        if field.name != "model_type" and field.name in text:
            effective[field.name] = deepcopy(text[field.name])
    effective["model_type"] = config["model_type"]
    if "embedding_dim" in config:
        effective["embedding_dim"] = config["embedding_dim"]
    else:
        effective.pop("embedding_dim", None)

    aliases = ("quantization", "quantization_config")
    for key in aliases:
        effective.pop(key, None)
    for source in (config, vlm, text):
        if any(key in source for key in aliases):
            effective.update(
                {key: deepcopy(source[key]) for key in aliases if key in source}
            )
            recipe = effective.get("quantization_config")
            if "quantization" not in effective:
                if recipe is None:
                    effective["quantization"] = None
                elif (
                    isinstance(recipe, dict) and {"bits", "group_size"} <= recipe.keys()
                ):
                    effective["quantization"] = deepcopy(recipe)
            break
    return effective


def _nemotron_loader_config(config: dict[str, Any]) -> dict[str, Any]:
    effective = deepcopy(config)
    text_defaults = {
        "model_type": "llama_bidirec",
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "intermediate_size": 8192,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "max_position_embeddings": 131072,
        "vocab_size": 128266,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
    }
    vision_defaults = {
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "num_channels": 3,
        "image_size": 512,
        "patch_size": 16,
        "layer_norm_eps": 1e-6,
    }
    text = effective.pop("llm_config", None) or effective.get("text_config") or {}
    effective["text_config"] = {**text_defaults, **text, "tie_word_embeddings": True}
    effective["vision_config"] = {
        **vision_defaults,
        **(effective.get("vision_config") or {}),
    }
    return effective


def _supports_prompt_exclusion_without_boundaries(config: dict[str, Any]) -> bool:
    flags = {
        key: value for key, value in config.items() if key.startswith("pooling_mode_")
    }
    if any(not isinstance(value, bool) for value in flags.values()):
        return False
    active = [key for key, value in flags.items() if value]
    if any(key not in _POOLING_FLAG_MODES for key in active) or len(active) > 1:
        return False
    declared = config.get("pooling_mode")
    if "pooling_mode" in config:
        if not isinstance(declared, str):
            return False
        mode = _POOLING_MODE_ALIASES.get(declared.strip().lower())
        if mode is None:
            return False
        matching = next(
            key for key, value in _POOLING_FLAG_MODES.items() if value == mode
        )
        if matching in flags and not flags[matching]:
            return False
        if active and active[0] != matching:
            return False
    elif len(active) != 1:
        return False
    return True


@dataclass
class EmbeddingOutput:
    """Output from embedding generation."""

    embeddings: List[List[float]]
    """List of embedding vectors, one per input text."""

    total_tokens: int
    """Total number of tokens in the input."""

    dimensions: int = 0
    """Dimension of each embedding vector."""


class _ChatTemplateEmbeddingInputs:
    """Build text, image, and audio embedding inputs with an mlx-vlm processor."""

    def __init__(
        self,
        processor,
        family: _ChatEmbeddingFamily,
        audio_max_seconds: float | None = None,
        max_input_length: int | None = None,
    ):
        self._processor = processor
        self.family = family
        self._max_length = max_input_length or family.context_length
        feature_extractor = getattr(processor, "feature_extractor", None)
        self._sampling_rate = getattr(feature_extractor, "sampling_rate", 16000)
        # The feature extractor cuts audio at its call-time max_length default
        # (Gemma 4: 30 s); only the call can raise it.
        self._audio_kwargs = (
            {"max_length": int(audio_max_seconds * self._sampling_rate)}
            if audio_max_seconds and audio_max_seconds > 0
            else {}
        )

    def prepare_embedding_inputs(
        self,
        inputs,
        return_tensors: str = "mlx",
        *,
        max_length: int | None = None,
        truncation: bool = True,
    ):
        family = self.family
        conversations = []
        for item in inputs:
            text = [{"type": "text", "text": item["text"]}] if item.get("text") else []
            media = []
            if item.get("image"):
                media.append(
                    {"type": "image", "image": load_image(item["image"], field="image")}
                )
            if item.get("audio"):
                # Decoded bytes from embed(); the processor takes the waveform
                # as-is and only loads str/Path values itself.
                waveform = load_audio(io.BytesIO(item["audio"]), sr=self._sampling_rate)
                media.append({"type": "audio", "audio": waveform})
            content = media + text if family.media_before_text else text + media
            messages = [{"role": "user", "content": content}]
            if family.system_instruction:
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": item.get("instruction")
                                or family.system_instruction,
                            }
                        ],
                    },
                )
            conversations.append(messages)
        has_audio = any(item.get("audio") for item in inputs)

        def prepare(messages, with_audio=False):
            processor_kwargs = {"padding": True, "truncation": False}
            if with_audio and self._audio_kwargs:
                processor_kwargs["audio_kwargs"] = self._audio_kwargs
            return self._processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors=return_tensors,
                add_generation_prompt=family.add_generation_prompt,
                processor_kwargs=processor_kwargs,
            )

        def lengths(batch, count):
            mask = batch.get("attention_mask")
            if mask is None:
                return [batch["input_ids"].shape[1]] * count
            return np.asarray(mask).sum(axis=1).tolist()

        batch = prepare(conversations, has_audio)
        if family.template_text and truncation:
            limit = max_length if max_length is not None else self._max_length
            counts = lengths(batch, len(conversations))
            changed = False
            for index, (item, count) in enumerate(zip(inputs, counts)):
                if item.get("image") or item.get("audio") or count <= limit:
                    continue
                conversation = conversations[index]
                original = item.get("text") or ""
                side = getattr(self._processor.tokenizer, "truncation_side", "right")

                def set_text(
                    size, original=original, side=side, conversation=conversation
                ):
                    if side == "left":
                        text = original[-size:] if size else ""
                    else:
                        text = original[:size]
                    conversation[-1]["content"] = (
                        [{"type": "text", "text": text}] if text else []
                    )

                set_text(0)
                overhead = lengths(prepare([conversation]), 1)[0]
                if overhead > limit:
                    raise ValueError(
                        f"Embedding chat template requires {overhead} tokens, "
                        f"which exceeds the {limit}-token limit"
                    )
                best = 0
                low, high = 1, len(original) - 1
                while low <= high:
                    size = (low + high) // 2
                    set_text(size)
                    if lengths(prepare([conversation]), 1)[0] <= limit:
                        best = size
                        low = size + 1
                    else:
                        high = size - 1
                boundary = len(original) - best if side == "left" else best
                if side == "left":
                    while boundary > 0 and original[boundary - 1].isalnum():
                        boundary -= 1
                    candidate = len(original) - boundary
                else:
                    while boundary < len(original) and original[boundary].isalnum():
                        boundary += 1
                    candidate = boundary
                if candidate > best:
                    set_text(candidate)
                    if lengths(prepare([conversation]), 1)[0] <= limit:
                        best = candidate
                set_text(best)
                changed = True
            if changed:
                batch = prepare(conversations, has_audio)
                counts = lengths(batch, len(conversations))
                if any(
                    count > limit and not (item.get("image") or item.get("audio"))
                    for item, count in zip(inputs, counts)
                ):
                    raise ValueError(
                        f"Embedding text exceeds the {limit}-token limit after rendering"
                    )
        # Truncation can cut media placeholders, so reject long inputs instead.
        length = batch["input_ids"].shape[1]
        if length > self._max_length:
            raise ValueError(
                f"Embedding input is {length} tokens, which exceeds the "
                f"{self._max_length}-token limit for media inputs"
            )
        return batch


class MLXEmbeddingModel:
    """
    Unified interface for embedding models on Apple's MLX framework.

    Supports:
    - Native XLMRoBERTa and BERT embedding
    - Native Qwen2-decoder embedding (last-token + L2; jina-code, gte-Qwen2)
    - mlx-vlm for every other embedding family, including EmbeddingGemma 2
      (text, image, and audio) and Qwen3-VL-Embedding (text and image)

    Example:
        >>> model = MLXEmbeddingModel("mlx-community/all-MiniLM-L6-v2-4bit")
        >>> output = model.embed(["Hello, world!", "How are you?"])
        >>> print(len(output.embeddings))  # 2
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
        audio_enabled: bool = False,
        audio_max_seconds: Optional[float] = None,
    ):
        """
        Initialize the MLX embedding model.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Allow execution of custom Python shipped inside
                the model repository. Off by default for security (issue #926).
            audio_enabled: Load the audio tower of models that have one, so
                audio items are accepted. Off by default to save memory.
            audio_max_seconds: Longest audio item read before the waveform is
                cut. None keeps the processor default (Gemma 4 audio: 30 s).
        """
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.audio_enabled = audio_enabled
        self.audio_max_seconds = audio_max_seconds

        self.model = None
        self.processor = None
        self._loaded = False
        self._hidden_size: Optional[int] = None
        self._using_native = False
        self._is_compiled = False
        self._compiled_embed = None
        self._pad_to_max_length = False
        self._pooling_mode: Optional[str] = None
        self._pooling_source: str = "not resolved"
        self._pooling_config: Optional[Dict[str, Any]] = None
        self._media_processor = None
        self._siglip_images: SiglipImageEncoder | None = None
        self._supports_audio = False
        self._right_pad_text = False
        self._prompt_exclusion_compatible = False

    # (hidden_size, num_hidden_layers) of Qwen3-Embedding-0.6B and -8B.
    _FP16_PROMOTE_SHAPES = {(1024, 28), (4096, 36)}

    def _should_promote_to_fp16(self) -> bool:
        """Match an unquantized Qwen3-Embedding 0.6B or 8B checkpoint.

        bf16 matmuls miss the 1e-3 fp32 conformance gate on these models
        (max |delta| 0.0037 vs 0.0006 in fp16). Other sizes are not validated.
        """
        try:
            with open(Path(self.model_name) / "config.json") as fh:
                cfg = json.load(fh)
        except (OSError, ValueError):
            return False
        if not isinstance(cfg, dict) or cfg.get("model_type") != "qwen3":
            return False
        if cfg.get("quantization") or cfg.get("quantization_config"):
            return False
        name = f"{self.model_name} {cfg.get('_name_or_path') or ''}".lower()
        if "qwen3-embedding" not in name:
            return False
        shape = (cfg.get("hidden_size"), cfg.get("num_hidden_layers"))
        return shape in self._FP16_PROMOTE_SHAPES

    def _promote_bf16_to_fp16(self, module: Any) -> None:
        if not self._should_promote_to_fp16():
            return
        params = module.parameters()
        if not any(v.dtype == mx.bfloat16 for _, v in tree_flatten(params)):
            return
        module.update(
            tree_map(
                lambda a: a.astype(mx.float16) if a.dtype == mx.bfloat16 else a,
                params,
            )
        )
        mx.eval(module.parameters())
        logger.info("Promoted bfloat16 parameters to float16 for %s", self.model_name)

    # Fallbacks for MLX conversions that dropped the sentence-transformers
    # metadata. Reviewed against the concrete checkpoints on the Hub: none of
    # the MLX BGE-M3 / Qwen3-Embedding / Qwen3-VL-Embedding repos ship
    # ``1_Pooling/config.json``, so detection alone never fires for exactly the
    # models this fix is meant to serve.
    _FAMILY_POOLING = (
        ("qwen3-vl-embedding", "lasttoken"),
        ("qwen3-embedding", "lasttoken"),
        ("bge-m3", "cls"),
    )

    def _pooling_config_path(self, model_path: Path) -> Optional[Path]:
        """Locate the Pooling module through ``modules.json``.

        sentence-transformers describes its pipeline there; the Pooling module
        carries its own directory in ``path`` (usually ``1_Pooling``, but not
        always). Hardcoding ``1_Pooling`` misses non-standard exports, and some
        MLX repos keep the ``modules.json`` entry while dropping the directory
        itself -- so the path is also checked for existence.
        """
        modules_path = model_path / "modules.json"
        if modules_path.is_file():
            try:
                with open(modules_path) as fh:
                    modules = json.load(fh)
                for module in modules if isinstance(modules, list) else []:
                    if "Pooling" in str(module.get("type", "")):
                        candidate = model_path / str(module.get("path", ""))
                        cfg = candidate / "config.json"
                        if cfg.is_file():
                            return cfg
            except (OSError, ValueError) as e:
                logger.debug(f"Could not read modules.json for {self.model_name}: {e}")
        # Fall back to the conventional location for exports without modules.json.
        conventional = model_path / "1_Pooling" / "config.json"
        return conventional if conventional.is_file() else None

    @staticmethod
    def _pooling_mode_from_config(cfg: Dict[str, Any]) -> Optional[str]:
        """Read either config dialect.

        Recent sentence-transformers releases write a single
        ``{"pooling_mode": "lasttoken"}`` string; older ones set one boolean per
        mode. Qwen3-VL-Embedding uses the newer form, so a boolean-only parser
        silently ignores a pooling config that is actually present.
        """
        declared = cfg.get("pooling_mode")
        if isinstance(declared, str):
            mode = _POOLING_MODE_ALIASES.get(declared.strip().lower())
            if mode:
                return mode
        for key, mode in (
            ("pooling_mode_cls_token", "cls"),
            ("pooling_mode_max_tokens", "max"),
            ("pooling_mode_lasttoken", "lasttoken"),
            ("pooling_mode_mean_tokens", "mean"),
        ):
            if cfg.get(key):
                return mode
        return None

    def _resolve_pooling_mode(
        self, model_path: Path | None = None
    ) -> tuple[str | None, str]:
        """Resolve the pooling mode and report where it came from.

        Returns ``(mode, source)``. ``source`` is logged at load time so an
        operator can tell a declared mode from an inferred one without reading
        the code -- silent pooling changes are exactly what makes retrieval
        quality drift unnoticed.
        """
        if model_path is None:
            model_path = Path(self.model_name)
        self._pooling_config = None
        cfg_path = self._pooling_config_path(model_path)
        if cfg_path is not None:
            try:
                with open(cfg_path) as fh:
                    cfg = json.load(fh)
                if isinstance(cfg, dict):
                    self._pooling_config = cfg
                    mode = self._pooling_mode_from_config(cfg)
                    if mode:
                        return mode, str(cfg_path.relative_to(model_path))
                    return None, str(cfg_path.relative_to(model_path))
            except (OSError, ValueError) as e:
                logger.debug(f"Could not read pooling config for {self.model_name}: {e}")

        haystack = str(self.model_name).lower()
        config_path = model_path / "config.json"
        if config_path.is_file():
            try:
                with open(config_path) as fh:
                    haystack += " " + str(json.load(fh).get("_name_or_path", "")).lower()
            except (OSError, ValueError):
                pass
        for needle, mode in self._FAMILY_POOLING:
            if needle in haystack:
                return mode, f"known family ({needle})"
        return None, "not declared"

    def _load_native(self) -> bool:
        """
        Try to load using native omlx implementations (xlm_roberta, bert).

        Returns True if native loading succeeded, False otherwise.
        """
        from transformers import AutoTokenizer

        model_path = Path(self.model_name)
        config_path = model_path / "config.json"
        if not config_path.exists():
            logger.debug(f"No config.json at {model_path}, native loading skipped")
            return False

        try:
            with open(config_path) as f:
                config_dict = json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.debug("Failed to read config.json, native loading skipped")
            return False

        architectures = config_dict.get("architectures", [])
        arch = architectures[0] if architectures else ""

        native_arch_modules = {
            "XLMRobertaModel": "xlm_roberta",
            "BertModel": "xlm_roberta",
            "BertForMaskedLM": "xlm_roberta",
            "Qwen2ForCausalLM": "qwen2_embedding",
        }
        module_name = native_arch_modules.get(arch)
        if module_name is None:
            logger.debug(
                f"Architecture '{arch}' not natively supported for embedding, "
                "trying mlx-vlm"
            )
            return False

        try:
            from importlib import import_module

            native_module = import_module(f"{__package__}.{module_name}")
            Model = native_module.Model
            ModelArgs = native_module.ModelArgs

            known_fields = {f.name for f in ModelArgs.__dataclass_fields__.values()}
            model_config = {
                k: v for k, v in config_dict.items() if k in known_fields
            }
            model_config["architectures"] = architectures

            config = ModelArgs(**model_config)
            model_instance = Model(config)

            weights = {}
            weight_files = list(model_path.glob("*.safetensors"))
            if not weight_files:
                logger.debug(f"No safetensors files found in {model_path}")
                return False

            for wf in weight_files:
                weights.update(mx.load(str(wf)))

            weights = model_instance.sanitize(weights)
            self._validate_native_weights(model_instance, weights)
            model_instance.load_weights(list(weights.items()), strict=False)
            mx.eval(model_instance.parameters())
            # Embedding inference must be deterministic: put the model in eval
            # mode so dropout (p>0 in XLM-RoBERTa/BERT) is disabled. Without this
            # every /v1/embeddings call applies random dropout, producing
            # non-deterministic, corrupted vectors.
            model_instance.train(False)

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    use_fast=False,
                    trust_remote_code=self.trust_remote_code,
                )
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    trust_remote_code=self.trust_remote_code,
                )

            self.model = model_instance
            self.processor = tokenizer
            self._hidden_size = config.hidden_size
            self._loaded = True
            self._using_native = True
            self._is_compiled = False
            self._compiled_embed = None
            logger.info(
                f"Embedding model loaded natively: {self.model_name} "
                f"(arch={arch}, hidden_size={config.hidden_size}, "
                f"pooling={self._pooling_mode}, source={self._pooling_source})"
            )
            return True

        except Exception as e:
            logger.debug(f"Native loading failed for {self.model_name}: {e}")
            return False

    def _load_mlx_vlm(self) -> None:
        """Load any embedding family mlx-vlm implements."""
        model_path = get_model_path(self.model_name)
        self._pooling_mode, self._pooling_source = self._resolve_pooling_mode(
            model_path
        )
        config = load_config(model_path)
        ensure_model_code_trusted(
            config,
            model_path=model_path,
            trust_remote_code=self.trust_remote_code,
        )
        # Checkpoints may spell model types with hyphens (gemma3-text), but
        # mlx-vlm resolves model modules by underscore names.
        model_type = str(config.get("model_type", "")).lower().replace("-", "_")
        model_type = _EMBEDDING_MODEL_TYPE_OVERRIDES.get(model_type, model_type)
        config["model_type"] = model_type
        original_config = config
        vlm_config = config.get("vlm_config")
        nested_colqwen = model_type == "colqwen2_5" and isinstance(vlm_config, dict)
        legacy_nemotron = model_type == "llama_nemotron_vl_embedding"
        if nested_colqwen or legacy_nemotron:
            pending = [config]
            while pending:
                nested = pending.pop()
                ensure_model_code_trusted(
                    nested,
                    model_path=model_path,
                    trust_remote_code=self.trust_remote_code,
                )
                pending.extend(
                    value for value in nested.values() if isinstance(value, dict)
                )
        if nested_colqwen:
            config = _colqwen_loader_config(config)
        elif legacy_nemotron:
            config = _nemotron_loader_config(config)
        if nested_colqwen or legacy_nemotron:
            ensure_model_code_trusted(
                config,
                model_path=model_path,
                trust_remote_code=self.trust_remote_code,
            )
        # The audio tower is resident even for text-only requests, so it loads
        # only when the model's embedding_audio_enabled setting asks for it.
        # mlx-vlm defaults a missing audio_config to {}, which builds the tower.
        if not self.audio_enabled:
            config["audio_config"] = None
        normalize_encoder_quantization_config(config)
        logger.info(f"Loading embedding model via mlx-vlm: {self.model_name}")
        model = load_embedding_model(model_path, config=config)
        if nested_colqwen:
            from mlx_vlm.models.colqwen2_5 import ModelConfig

            text = vlm_config.get("text_config")
            text = text if isinstance(text, dict) else {}
            text_type = text.get("model_type") or vlm_config.get("model_type")
            if text_type is not None:
                model.config.text_config.model_type = text_type
                model.language_model.model_type = text_type
            model.config.vocab_size = original_config.get(
                "vocab_size",
                vlm_config.get(
                    "vocab_size", ModelConfig.__dataclass_fields__["vocab_size"].default
                ),
            )
        elif legacy_nemotron:
            set_siglip_activation(model.vision_model, "precise")
        processor = load_processor(
            model_path,
            add_detokenizer=False,
            trust_remote_code=self.trust_remote_code,
        )
        if model_type == "siglip":
            vision = load_siglip_vision(model_path, config)
            image_processor = getattr(processor, "image_processor", None)
            if vision is not None and image_processor is not None:
                self._siglip_images = SiglipImageEncoder(vision, image_processor)
        text_processor = (
            processor
            if self._uses_custom_embedding_inputs(processor)
            else getattr(processor, "tokenizer", processor)
        )
        family = _CHAT_EMBEDDING_FAMILIES.get(model_type)
        if self._pooling_config is not None:
            pooling_config = dict(self._pooling_config)
            if (
                pooling_config.get("include_prompt") is False
                and _supports_prompt_exclusion_without_boundaries(pooling_config)
                and not any(
                    pooling_config.get(key) is not None
                    for key in ("prompt_length", "prompt_mask", "prompt_token_mask")
                )
                and family is None
            ):
                pooling_config["include_prompt"] = True
                self._prompt_exclusion_compatible = True
            # mlx-vlm matches string modes exactly, while omlx accepts aliases.
            declared = pooling_config.get("pooling_mode")
            if isinstance(declared, str):
                mode = _POOLING_MODE_ALIASES.get(declared.strip().lower())
                if mode:
                    pooling_config["pooling_mode"] = mode
            model.pooling_config = pooling_config
        elif self._pooling_mode:
            model.pooling_config = {"pooling_mode": self._pooling_mode}
        patch_modernbert_attention(model)
        self._promote_bf16_to_fp16(model)

        self.model = model
        # Text requests use the tokenizer path. Multimodal processors take
        # images as their first positional argument.
        self.processor = text_processor
        if family is not None:
            context_length = family.context_length
            if model_type == "qwen3_vl_embedding":
                context_length = (
                    self._context_length_from_config(model.config) or context_length
                )
            model.max_input_length = context_length
            self._media_processor = _ChatTemplateEmbeddingInputs(
                processor, family, self.audio_max_seconds, context_length
            )
        self._supports_audio = getattr(model, "audio_tower", None) is not None
        self._pad_to_max_length = model_type in _PAD_TO_MAX_LENGTH_MODEL_TYPES
        self._right_pad_text = model_type == "colqwen2_5"
        if self._pad_to_max_length:
            model.max_input_length = model.config.text_config.max_position_embeddings
        self._hidden_size = self._embedding_dimension(model.config)
        self._using_native = False
        self._is_compiled = self._try_compile()
        self._loaded = True
        logger.info(
            f"Embedding model loaded via mlx-vlm: {self.model_name} "
            f"(model_type={model_type}, dimensions={self._hidden_size}, "
            f"compiled={self._is_compiled}, pooling={self._pooling_mode}, "
            f"source={self._pooling_source})"
        )

    @staticmethod
    def _embedding_dimension(config: Any) -> int | None:
        text_config = getattr(config, "text_config", None)
        for source, attr in (
            (text_config, "embedding_dim"),
            (config, "embedding_dim"),
            (config, "hidden_size"),
            (text_config, "hidden_size"),
        ):
            value = getattr(source, attr, None)
            if isinstance(value, int):
                return value
        return None

    def load(self) -> None:
        """Load the model and processor/tokenizer."""
        if self._loaded:
            return

        self._pooling_mode, self._pooling_source = self._resolve_pooling_mode()
        if self._load_native():
            return

        try:
            self._load_mlx_vlm()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No safetensors weight files found for '{self.model_name}'. "
                f"Embedding models require weights in safetensors format. "
                f"If this is a PyTorch model, use an MLX-converted version "
                f"(e.g., from mlx-community on HuggingFace)."
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _extract_embeddings_array(self, outputs, attention_mask=None):
        """Extract embedding tensor from model outputs as a 2D (batch, hidden) array."""
        # Honour the checkpoint's own declaration first. Some MLX conversions of
        # sentence-transformers models expose a mean-pooled ``text_embeds`` while
        # the checkpoint was trained for CLS pooling (e.g. BGE-M3), which silently
        # degrades retrieval instead of failing. Mode is resolved once at load().
        #
        # Pooling goes through the existing mask-aware helpers: a bare
        # ``[:, -1]`` is only correct under left padding and an unmasked
        # ``mean(axis=1)`` averages pad tokens in, both of which corrupt vectors
        # in mixed-length batches while single inputs still look fine. The
        # result is L2-normalized like every other path out of this method.
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if (
            isinstance(self.model, (LlamaBidirec, LlamaNemotron))
            and self._pooling_mode
            and last_hidden is not None
            and last_hidden.ndim == 3
        ):
            mask = attention_mask
            if mask is None:
                mask = mx.ones(last_hidden.shape[:2], dtype=mx.int32)
            return normalize_embeddings(
                pool_by_config(last_hidden, mask, self.model.pooling_config)
            )
        if (
            (self._using_native or isinstance(self.model, Qwen3VLEmbedding))
            and self._pooling_mode
            and last_hidden is not None
            and last_hidden.ndim == 3
        ):
            if self._pooling_mode == "cls":
                if (
                    isinstance(self.model, Qwen3VLEmbedding)
                    and attention_mask is not None
                ):
                    return normalize_embeddings(
                        cls_pooling(last_hidden, attention_mask)
                    )
                return normalize_embeddings(last_hidden[:, 0, :])
            if self._pooling_mode == "lasttoken":
                return normalize_embeddings(
                    last_token_pool(last_hidden, attention_mask)
                )
            if self._pooling_mode == "mean" and attention_mask is not None:
                return normalize_embeddings(
                    mean_pooling(last_hidden, attention_mask)
                )

        is_image = False
        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            embeddings = outputs.text_embeds
        elif getattr(outputs, "image_embeds", None) is not None:
            embeddings = outputs.image_embeds
            is_image = True
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        elif (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            embeddings = outputs.last_hidden_state
        else:
            raise ValueError(
                "Model output does not contain expected embedding fields "
                "(text_embeds, pooler_output, or last_hidden_state)"
            )

        # Some models (e.g. ModernBERT loaded as MaskedLM) skip pooling and return
        # per-token features with shape (batch, seq_len, hidden). Mean pool so
        # /v1/embeddings always emits one vector per input.
        if embeddings.ndim == 3:
            if attention_mask is not None and not is_image:
                if not isinstance(attention_mask, mx.array):
                    attention_mask = mx.array(attention_mask)
                if (
                    attention_mask.ndim != 2
                    or attention_mask.shape != embeddings.shape[:2]
                ):
                    raise ValueError(
                        "Attention mask shape does not match the token embeddings"
                    )
                mask = attention_mask.astype(mx.bool_)[:, :, None]
                values = mx.where(mask, embeddings, mx.zeros_like(embeddings))
                count = mx.maximum(mx.sum(mask.astype(mx.float32), axis=1), 1)
                embeddings = mx.sum(values, axis=1) / count
            else:
                embeddings = mx.mean(embeddings, axis=1)
        return embeddings

    def _validate_native_weights(
        self, model_instance, weights: Dict[str, Any]
    ) -> None:
        """Reject native checkpoints with missing or shape-incompatible core weights."""
        expected_weights = dict(tree_flatten(model_instance.parameters()))
        expected_weight_names = set(expected_weights.keys())
        provided_weight_names = set(weights.keys())
        missing_weight_names = expected_weight_names - provided_weight_names

        optional_missing_prefixes = ("pooler.",)
        required_missing = sorted(
            name
            for name in missing_weight_names
            if not name.startswith(optional_missing_prefixes)
        )
        if required_missing:
            preview = ", ".join(required_missing[:10])
            suffix = "..." if len(required_missing) > 10 else ""
            raise ValueError(
                "Native embedding checkpoint is missing required weights: "
                f"{preview}{suffix}"
            )

        shape_mismatches = []
        for name in expected_weight_names & provided_weight_names:
            expected_shape = tuple(expected_weights[name].shape)
            provided_shape = tuple(weights[name].shape)
            if expected_shape != provided_shape:
                shape_mismatches.append((name, expected_shape, provided_shape))

        if shape_mismatches:
            preview = ", ".join(
                f"{name}: expected {expected_shape}, got {provided_shape}"
                for name, expected_shape, provided_shape in shape_mismatches[:5]
            )
            suffix = "..." if len(shape_mismatches) > 5 else ""
            raise ValueError(
                "Native embedding checkpoint has incompatible weight shapes: "
                f"{preview}{suffix}"
            )

    def _uses_custom_embedding_inputs(self, processor) -> bool:
        """Return True when processor exposes a custom embedding input API."""
        for attr_name in ("prepare_embedding_inputs", "prepare_model_inputs"):
            try:
                inspect.getattr_static(processor, attr_name)
                return True
            except AttributeError:
                continue
        return False

    def _normalize_embedding_inputs(
        self,
        inputs: Union[str, Dict[str, str], List[str], List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        """Normalize embedding inputs into item dicts."""
        if not inputs:
            return []
        if isinstance(inputs, str):
            return [{"text": inputs}]
        if isinstance(inputs, dict):
            return [dict(inputs)]
        first = inputs[0]
        if isinstance(first, str):
            return [{"text": text} for text in inputs]
        return [dict(item) for item in inputs]

    @staticmethod
    def _positive_context_length(value: Any) -> Optional[int]:
        """Return a usable positive context length from config/tokenizer metadata."""
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        if 0 < value < _TOKENIZER_MAX_LENGTH_SENTINEL:
            return value
        return None

    @classmethod
    def _get_config_value(cls, config: Any, key: str) -> Optional[int]:
        if config is None:
            return None
        if isinstance(config, dict):
            return cls._positive_context_length(config.get(key))
        return cls._positive_context_length(getattr(config, key, None))

    @classmethod
    def _context_length_from_config(cls, config: Any) -> Optional[int]:
        """Read context length from model config objects or dictionaries."""
        for key in _CONTEXT_LENGTH_ATTRS:
            value = cls._get_config_value(config, key)
            if value is not None:
                return value

        for nested_key in ("text_config", "language_config"):
            nested = None
            if isinstance(config, dict):
                nested = config.get(nested_key)
            elif config is not None:
                nested = getattr(config, nested_key, None)
            for key in _CONTEXT_LENGTH_ATTRS:
                value = cls._get_config_value(nested, key)
                if value is not None:
                    return value

        return None

    def _resolve_max_length(self, max_length: int | None) -> int:
        """Resolve embedding token length when callers omit an explicit limit."""
        if max_length is not None:
            value = self._positive_context_length(max_length)
            if value is None:
                raise ValueError("max_length must be a positive integer")
            return value

        if (
            isinstance(self._media_processor, _ChatTemplateEmbeddingInputs)
            and self._media_processor.family.template_text
        ):
            return self._media_processor.family.context_length

        for config in (
            getattr(self.model, "config", None),
            getattr(self.processor, "config", None),
        ):
            value = self._context_length_from_config(config)
            if value is not None:
                return value

        processor = self.processor
        tokenizers = [
            processor,
            getattr(processor, "tokenizer", None),
            getattr(processor, "_tokenizer", None),
        ]
        for tokenizer in tokenizers:
            for attr_name in ("model_max_length", "max_length"):
                value = self._positive_context_length(
                    getattr(tokenizer, attr_name, None)
                )
                if value is not None:
                    return value

        return _DEFAULT_EMBEDDING_MAX_LENGTH

    def _prepare_embedding_inputs(
        self,
        processor,
        inputs: Union[List[str], List[Dict[str, str]]],
        max_length: int,
        padding: bool,
        truncation: bool,
    ):
        """Prepare inputs for embedding inference."""
        normalized_inputs = self._normalize_embedding_inputs(inputs)

        if isinstance(processor, _ChatTemplateEmbeddingInputs):
            return processor.prepare_embedding_inputs(
                normalized_inputs,
                return_tensors="mlx",
                max_length=max_length,
                truncation=truncation,
            )

        if self._uses_custom_embedding_inputs(processor):
            if hasattr(processor, "prepare_embedding_inputs"):
                prepared = processor.prepare_embedding_inputs(
                    normalized_inputs, return_tensors="mlx"
                )
            else:
                prepared = processor.prepare_model_inputs(
                    normalized_inputs, return_tensors="mlx"
                )
            if self._prompt_exclusion_compatible and any(
                prepared.get(key) is not None
                for key in ("prompt_length", "prompt_mask", "prompt_token_mask")
            ):
                raise NotImplementedError(
                    "Explicit prompt boundaries are not supported for pooling"
                )
            return prepared

        if any("image" in item for item in normalized_inputs):
            raise ValueError(
                f"Embedding model '{self.model_name}' does not support image inputs"
            )

        encoded = processor(
            [item.get("text", "") for item in normalized_inputs],
            padding="max_length" if self._pad_to_max_length else padding,
            truncation=truncation,
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="np",
        )
        if self._pad_to_max_length:
            encoded = self._pad_to_position_table(encoded, processor, truncation)
        if self._right_pad_text:
            if "attention_mask" not in encoded:
                raise ValueError("ColQwen text inputs require an attention mask")
            mask = np.asarray(encoded["attention_mask"])
            order = np.argsort(mask == 0, axis=1, kind="stable")
            for key in ("input_ids", "attention_mask", "token_type_ids"):
                if key in encoded:
                    encoded[key] = np.take_along_axis(
                        np.asarray(encoded[key]), order, axis=1
                    )
        return {
            key: mx.array(encoded[key])
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in encoded
        }

    def _pad_to_position_table(
        self, encoded, processor, truncation: bool
    ) -> dict[str, np.ndarray]:
        """Right-pad to the full position table after request truncation.

        SigLIP pools the last position and ignores the attention mask, so its
        embedding is only correct when padded to the length it was trained on.
        """
        target = self.model.max_input_length
        pad_values = {
            "input_ids": getattr(processor, "pad_token_id", None) or 0,
            "attention_mask": 0,
            "token_type_ids": 0,
        }
        padded = {}
        for key, values in encoded.items():
            values = np.asarray(values)
            extra = target - values.shape[-1]
            if extra < 0:
                if not truncation:
                    raise ValueError(
                        f"SigLIP input length {values.shape[-1]} exceeds its "
                        f"position table limit of {target} with truncation disabled"
                    )
                values = values[:, :target]
                extra = 0
            if key in pad_values and extra > 0:
                values = np.pad(
                    values, ((0, 0), (0, extra)), constant_values=pad_values[key]
                )
            padded[key] = values
        return padded

    def _forward(self, inputs: dict[str, Any]):
        """Run the mlx-vlm model on prepared inputs.

        Qwen3-VL caches rope positions without accounting for a new batch's
        mask. Clear the cache around each call, including after compile traces.
        """
        language_model = getattr(self.model, "language_model", None)
        resets_rope = hasattr(language_model, "_position_ids")
        if resets_rope:
            language_model._position_ids = None
            language_model._rope_deltas = None
        try:
            if isinstance(self.model, Lfm2Colbert):
                hidden, _ = self.model._encode(
                    inputs["input_ids"],
                    inputs.get("attention_mask"),
                    mask_convolutions=True,
                )
                projected = self.model.projection(hidden).astype(mx.float32)
                return VlmEmbeddingOutput(
                    last_hidden_state=hidden,
                    text_embeds=normalize_token_embeddings(projected),
                )
            return self.model(**inputs)
        finally:
            if resets_rope:
                language_model._position_ids = None
                language_model._rope_deltas = None

    def _try_compile(self) -> bool:
        """
        Compile a primitive-output embedding forward function.

        Root-cause fix:
        - Compiling model.__call__ directly can return arrays without primitives
          for some embedding/reranker models, causing eval() runtime errors.
        - We compile a narrower function that returns only the final embedding array.
        """
        try:

            def _compiled_embed(inputs):
                outputs = self._forward(inputs)
                # The mask travels with the request; mask-aware pooling needs it.
                return self._extract_embeddings_array(
                    outputs, inputs.get("attention_mask")
                )

            self._compiled_embed = mx.compile(_compiled_embed)

            # Real requests arrive with a traced attention_mask (the tokenizer
            # path always emits one). A mask-less probe lets a model whose
            # forward branches on the mask pass here — its internally-built
            # default mask is a tracing constant — and then the first real
            # request trips the traced-mask branch and silently disables
            # compile for the rest of the process (issue #2447). The reranker
            # probe already includes the mask.
            test_inputs = {
                "input_ids": mx.zeros((1, 4), dtype=mx.int32),
                "attention_mask": mx.ones((1, 4), dtype=mx.int32),
            }
            _ = self._compiled_embed(test_inputs)

            logger.info(
                f"mx.compile enabled for {self.model_name} "
                f"(primitive embedding path)"
            )
            return True
        except Exception as e:
            logger.info(f"mx.compile unavailable for {self.model_name}: {e}")
            self._compiled_embed = None
            return False

    def close(self) -> None:
        """Release model, processor, and compiled embedding resources."""
        if not self._loaded and self.model is None and self.processor is None:
            self._compiled_embed = None
            self._is_compiled = False
            return

        logger.info(
            "Releasing embedding model resources: %s "
            "(compiled=%s, native=%s)",
            self.model_name,
            self._is_compiled,
            self._using_native,
        )

        self._compiled_embed = None
        self._is_compiled = False

        self.model = None
        self.processor = None
        self._media_processor = None
        self._siglip_images = None
        self._supports_audio = False
        self._hidden_size = None
        self._loaded = False
        self._using_native = False
        self._pad_to_max_length = False
        self._right_pad_text = False
        self._prompt_exclusion_compatible = False

        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        gc.collect()

    def embed(
        self,
        inputs: Union[str, List[str], List[Dict[str, str]]],
        max_length: int | None = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> EmbeddingOutput:
        """
        Generate embeddings for input texts.

        Args:
            texts: List of input texts
            max_length: Maximum token length for each text. If omitted, use
                model/tokenizer metadata when available.
            padding: Whether to pad shorter sequences
            truncation: Whether to truncate longer sequences

        Returns:
            EmbeddingOutput with embeddings and token count
        """
        if not self._loaded:
            self.load()

        max_length = self._resolve_max_length(max_length)
        # Absolute position tables read out of range without an error.
        position_limit = getattr(self.model, "max_input_length", None)
        if isinstance(position_limit, int):
            max_length = min(max_length, position_limit)
        normalized_inputs = self._normalize_embedding_inputs(inputs)
        for item in normalized_inputs:
            if self._prompt_exclusion_compatible and any(
                item.get(key) is not None
                for key in ("prompt_length", "prompt_mask", "prompt_token_mask")
            ):
                raise NotImplementedError(
                    "Explicit prompt boundaries are not supported for pooling"
                )
            image_ref = item.get("image")
            if isinstance(image_ref, str):
                item["image"] = validate_image_data_uri(
                    image_ref,
                    field="items[].image",
                )
            audio_ref = item.get("audio")
            if isinstance(audio_ref, str):
                item["audio"] = _decode_input_audio_data(
                    audio_ref,
                    field="items[].audio",
                )
        input_texts = [item["text"] for item in normalized_inputs if "text" in item]
        has_image_inputs = any("image" in item for item in normalized_inputs)
        has_audio_inputs = any("audio" in item for item in normalized_inputs)
        if has_audio_inputs and not self._supports_audio:
            hint = (
                " (enable embedding_audio_enabled in its model settings)"
                if self._media_processor is not None and not self.audio_enabled
                else ""
            )
            raise ValueError(
                f"Embedding model '{self.model_name}' does not support audio "
                f"inputs{hint}"
            )

        processor = self.processor
        media_processor = self._media_processor
        if media_processor is not None and (
            has_image_inputs or has_audio_inputs or media_processor.family.template_text
        ):
            processor = media_processor
        uses_custom_embedding_inputs = self._uses_custom_embedding_inputs(processor)

        siglip_images = self._siglip_images if has_image_inputs else None
        if (
            has_image_inputs
            and siglip_images is None
            and (self._using_native or not uses_custom_embedding_inputs)
        ):
            raise ValueError(
                f"Embedding model '{self.model_name}' does not support image inputs"
            )

        batches = [list(range(len(normalized_inputs)))]
        if siglip_images is not None:
            if any(item.get("text") and "image" in item for item in normalized_inputs):
                raise ValueError(
                    "SigLIP embeds text and images separately; send each as its own item"
                )
            text_rows = [
                i for i, item in enumerate(normalized_inputs) if "image" not in item
            ]
            image_rows = [
                i for i, item in enumerate(normalized_inputs) if "image" in item
            ]
            batches = [rows for rows in (text_rows, image_rows) if rows]
        elif (
            not uses_custom_embedding_inputs
            and 1 < len(input_texts) == len(normalized_inputs)
            and len(input_texts) * max_length > ENCODER_BATCH_TOKEN_BUDGET
        ):
            lengths = self._text_token_lengths(
                processor, input_texts, max_length, truncation
            )
            batches = token_budget_batches(lengths, ENCODER_BATCH_TOKEN_BUDGET)

        positions: list[int] = []
        rows: list[list[float]] = []
        total_tokens: int | None = 0
        for batch in batches:
            batch_inputs = [normalized_inputs[i] for i in batch]
            if siglip_images is not None and "image" in batch_inputs[0]:
                embeddings_array = siglip_images.embed(batch_inputs)
                batch_tokens = 0
            else:
                embeddings_array, batch_tokens = self._embed_batch(
                    processor,
                    batch_inputs,
                    max_length,
                    padding,
                    truncation,
                    uses_custom_embedding_inputs,
                )
            # Evaluate per batch so peak memory stays at one batch.
            mx.eval(embeddings_array)
            positions.extend(batch)
            rows.extend(embeddings_array.tolist())
            if total_tokens is not None and batch_tokens is not None:
                total_tokens += batch_tokens
            else:
                total_tokens = None
        embeddings = [
            row for _, row in sorted(zip(positions, rows), key=lambda pair: pair[0])
        ]
        if siglip_images is not None and any(
            len(row) != len(embeddings[0]) for row in embeddings
        ):
            raise ValueError(
                "SigLIP text and image embeddings have different dimensions"
            )
        if total_tokens is None:
            total_tokens = self._count_tokens(normalized_inputs, max_length, truncation)
        dimensions = len(embeddings[0]) if embeddings else 0

        return EmbeddingOutput(
            embeddings=embeddings,
            total_tokens=total_tokens,
            dimensions=dimensions,
        )

    def _text_token_lengths(
        self, processor, texts: list[str], max_length: int, truncation: bool
    ) -> list[int]:
        """Return the token count of each text as the batch tokenizer sees it."""
        if callable(processor):
            encoded = processor(
                texts, padding=False, truncation=truncation, max_length=max_length
            )
            return [len(ids) for ids in encoded["input_ids"]]
        lengths = [
            len(processor.encode(text, add_special_tokens=True).ids) for text in texts
        ]
        return [min(n, max_length) for n in lengths] if truncation else lengths

    def _embed_batch(
        self,
        processor,
        normalized_inputs: list[dict[str, str]],
        max_length: int,
        padding: bool,
        truncation: bool,
        uses_custom_embedding_inputs: bool,
    ) -> tuple[mx.array, int | None]:
        """Run one padded batch and return its embeddings and token count."""
        embeddings_array = None
        total_tokens: int | None = None
        input_texts = [item["text"] for item in normalized_inputs if "text" in item]

        if self._using_native:
            if hasattr(processor, "__call__"):
                encoded = processor(
                    input_texts,
                    padding=padding,
                    truncation=truncation,
                    max_length=max_length,
                    return_tensors="np",
                )
                input_ids = mx.array(encoded["input_ids"])
                attention_mask = mx.array(encoded["attention_mask"])
            else:
                encoded_ids = []
                masks = []
                for text in input_texts:
                    enc = processor.encode(text, add_special_tokens=True)
                    ids = list(enc.ids)
                    if truncation:
                        ids = ids[:max_length]
                    encoded_ids.append(ids)
                max_len = max(len(ids) for ids in encoded_ids)
                padded = []
                for ids in encoded_ids:
                    pad_len = max_len - len(ids)
                    padded.append(ids + [0] * pad_len)
                    masks.append([1] * len(ids) + [0] * pad_len)
                input_ids = mx.array(padded)
                attention_mask = mx.array(masks)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings_array = self._extract_embeddings_array(
                outputs, attention_mask
            )
            total_tokens = self._count_prepared_tokens(
                {"attention_mask": attention_mask, "input_ids": input_ids}
            )
        else:
            # Media batches stay eager: the media merge calls .item(), and a
            # compile failure here would disable compile for text requests too.
            if (
                self._is_compiled
                and self._compiled_embed is not None
                and processor is not self._media_processor
            ):
                try:
                    inputs = self._prepare_embedding_inputs(
                        processor,
                        normalized_inputs,
                        max_length,
                        padding,
                        truncation,
                    )
                    if not isinstance(inputs, dict):
                        inputs = dict(inputs)
                    total_tokens = self._count_prepared_tokens(inputs)
                    embeddings_array = self._compiled_embed(inputs)
                except Exception as e:
                    logger.warning(
                        f"compiled embedding path failed for {self.model_name}: {e}; "
                        "disabling compile and falling back to eager forward()"
                    )
                    self._is_compiled = False
                    self._compiled_embed = None
                    total_tokens = None

            if embeddings_array is None:
                inputs = dict(
                    self._prepare_embedding_inputs(
                        processor,
                        normalized_inputs,
                        max_length,
                        padding,
                        truncation,
                    )
                )
                outputs = self._forward(inputs)
                total_tokens = self._count_prepared_tokens(inputs)
                embeddings_array = self._extract_embeddings_array(
                    outputs, inputs.get("attention_mask")
                )

        return embeddings_array, total_tokens

    def _count_tokens(
        self,
        inputs: Union[List[str], List[Dict[str, str]]],
        max_length: int,
        truncation: bool,
    ) -> int:
        """Count the input text tokens the model ran on."""
        total = 0
        processor = self.processor

        for item in self._normalize_embedding_inputs(inputs):
            text = item.get("text")
            if not text:
                continue
            if hasattr(processor, "encode"):
                tokens = processor.encode(text, add_special_tokens=True)
                if isinstance(tokens, list):
                    count = len(tokens)
                elif hasattr(tokens, "shape"):
                    count = tokens.shape[-1] if tokens.ndim > 0 else 1
                elif hasattr(tokens, "ids"):
                    count = len(tokens.ids)
                else:
                    count = len(tokens)
            elif hasattr(processor, "tokenizer"):
                tokens = processor.tokenizer.encode(text, add_special_tokens=True)
                count = len(tokens) if isinstance(tokens, list) else len(list(tokens))
            elif hasattr(processor, "_tokenizer"):
                tokens = processor._tokenizer.encode(text, add_special_tokens=True)
                count = len(tokens) if isinstance(tokens, list) else len(list(tokens))
            else:
                count = len(text.split()) + 2
            total += min(count, max_length) if truncation else count

        return total

    def _count_prepared_tokens(self, prepared_inputs: Dict[str, Any]) -> int:
        """Count tokens from prepared model inputs, including multimodal tokens."""
        attention_mask = prepared_inputs.get("attention_mask")
        if attention_mask is not None:
            try:
                return int(mx.sum(attention_mask).item())
            except (TypeError, ValueError):
                pass
            if isinstance(attention_mask, list):
                return int(sum(sum(row) if isinstance(row, list) else row for row in attention_mask))
            if hasattr(attention_mask, "tolist"):
                values = attention_mask.tolist()
                if values and isinstance(values[0], list):
                    return int(sum(sum(row) for row in values))
                return int(sum(values))

        input_ids = prepared_inputs.get("input_ids")
        if input_ids is None:
            return 0
        if hasattr(input_ids, "shape"):
            if len(input_ids.shape) == 0:
                return 1
            if len(input_ids.shape) == 1:
                return int(input_ids.shape[0])
            return int(input_ids.shape[0] * input_ids.shape[1])
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return int(sum(len(row) for row in input_ids))
            return int(len(input_ids))
        return 0

    @property
    def hidden_size(self) -> Optional[int]:
        """Get the embedding dimension."""
        return self._hidden_size

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._loaded:
            return {"loaded": False, "model_name": self.model_name}

        info = {
            "loaded": True,
            "model_name": self.model_name,
            "hidden_size": self._hidden_size,
            "native_implementation": self._using_native,
            "compiled": self._is_compiled,
        }

        if hasattr(self.model, "config"):
            config = self.model.config
            info.update(
                {
                    "model_type": getattr(config, "model_type", None),
                    "vocab_size": getattr(config, "vocab_size", None),
                    "max_position_embeddings": getattr(
                        config, "max_position_embeddings", None
                    ),
                }
            )

        return info

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        impl = "native" if self._using_native else "mlx-vlm"
        return (
            f"<MLXEmbeddingModel model={self.model_name} "
            f"status={status} impl={impl}>"
        )
