# SPDX-License-Identifier: Apache-2.0
"""
VLM (Vision-Language Model) engine with continuous batching.

This engine extends BatchedEngine to support vision-language models via
mlx-vlm. It provides:

- Image input processing (URL, base64, local file)
- Multi-image chat support
- Pre-computed vision embeddings for efficient batched inference
- Full compatibility with oMLX's tiered KV cache and boundary snapshots

Architecture:
    1. Images are extracted from messages and loaded as PIL Images
    2. mlx-vlm's prepare_inputs() tokenizes text and preprocesses images
    3. model.get_input_embeddings() runs vision encoder + embedding merge
    4. VLMModelAdapter receives pre-computed embeddings for prefill injection
    5. After prefill, decode uses standard token IDs (vision context in KV cache)

Usage:
    Engine is automatically selected when model_discovery detects a VLM model
    (engine_type="vlm"). No changes needed for API callers — the OpenAI
    vision API format is transparently handled.
"""

import asyncio
import copy
import logging
from collections.abc import AsyncIterator
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_special_tokens, detect_and_strip_partial
from ..cache.vision_feature_cache import VisionFeatureSSDCache
from ..models.vlm import VLMModelAdapter
from ..utils.image import (
    compute_image_hash,
    extract_images_from_messages,
)
from ..utils.tokenizer import get_tokenizer_config
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)

# OCR model types that require special handling.
OCR_MODEL_TYPES = {"deepseekocr", "deepseekocr_2", "dots_ocr", "glm_ocr"}

# OCR model types and their default markdown conversion prompts.
# When an OCR model receives a generic user prompt with an image,
# the prompt is automatically adjusted for markdown output.
OCR_MODEL_PROMPTS: Dict[str, str] = {
    "deepseekocr": "Convert the document to markdown.",
    "deepseekocr_2": "Convert the document to markdown.",
    "dots_ocr": "Convert this page to clean Markdown while preserving reading order.",
    "glm_ocr": "Text Recognition:",
}

# Extra stop sequences for OCR models to prevent degeneration.
# Many OCR models lack proper EOS handling and generate chat-turn
# tokens (<|user|>, <|im_start|>, etc.) indefinitely after the OCR output.
OCR_EXTRA_STOP_SEQUENCES: List[str] = [
    "<|user|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|endofassistant|>",
]

# Per-model OCR generation defaults from official configs.
# Applied automatically when no explicit user override is provided.
OCR_MODEL_GENERATION_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "glm_ocr": {
        "temperature": 0.0,
        "repetition_penalty": 1.1,
        "max_tokens": 4096,
    },
    "deepseekocr": {
        "temperature": 0.0,
        "max_tokens": 8192,
    },
    "deepseekocr_2": {
        "temperature": 0.0,
        "max_tokens": 8192,
    },
    "dots_ocr": {
        "temperature": 0.0,
        "max_tokens": 8192,
    },
}

_video_processor_patched = False
_gemma4_batched_decode_patched = False


def _patch_video_processor_bug():
    """Remove video_processor from transformers' auto-processor mapping.

    oMLX does not support video input. Without torchvision, transformers'
    AutoVideoProcessor crashes when loading VLM processors that have a
    video_preprocessor_config.json. By removing ``video_processor`` from
    the mapping, ``ProcessorMixin.get_attributes()`` no longer recognises
    it as a sub-processor and ``_get_arguments_from_pretrained`` never
    attempts to load it.
    """
    global _video_processor_patched
    if _video_processor_patched:
        return

    try:
        from transformers.processing_utils import MODALITY_TO_AUTOPROCESSOR_MAPPING

        mapping = MODALITY_TO_AUTOPROCESSOR_MAPPING._MAPPING_NAMES
        if "video_processor" in mapping:
            del mapping["video_processor"]
            logger.debug("Removed video_processor from MODALITY_TO_AUTOPROCESSOR_MAPPING")

        _video_processor_patched = True
    except (ImportError, AttributeError):
        pass


def _patch_gemma4_batched_decode():
    """Patch mlx-vlm's gemma4 model for correct batched decode.

    mlx-vlm's gemma4 reads shared KVs via cache.state which breaks in
    batched mode. This patch replaces it with mlx-lm's approach: explicit
    shared_kv/offset passing through intermediates[].

    Also patches ProportionalRoPE to handle per-element array offsets
    needed for batched decode with different prompt lengths.
    """
    global _gemma4_batched_decode_patched
    if _gemma4_batched_decode_patched:
        return

    try:
        from mlx_vlm.models.gemma4.language import (
            Attention,
            DecoderLayer,
            Gemma4TextModel,
            LanguageModel,
            scaled_dot_product_attention,
        )
        from mlx_vlm.models.gemma4.rope_utils import ProportionalRoPE

        # ── 1. Patch ProportionalRoPE for per-element array offsets ──

        _orig_rope = ProportionalRoPE.__call__

        def _patched_rope(self, x, offset=0):
            if isinstance(offset, mx.array) and offset.size > 1:
                parts = []
                for i in range(offset.size):
                    parts.append(
                        _orig_rope(self, x[i : i + 1], offset=int(offset[i].item()))
                    )
                return mx.concatenate(parts, axis=0)
            return _orig_rope(self, x, offset=offset)

        ProportionalRoPE.__call__ = _patched_rope

        # ── 2. Patch Attention.__call__ to pass shared_kv/offset ──

        def _patched_attn(self, x, mask=None, cache=None, shared_kv=None, offset=None):
            B, L, _ = x.shape

            queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
            queries = self.q_norm(queries)

            if shared_kv is not None:
                keys, values = shared_kv
            else:
                if offset is None:
                    offset = cache.offset if cache is not None else 0

                keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
                if self.use_k_eq_v:
                    values = keys
                else:
                    values = self.v_proj(x).reshape(
                        B, L, self.n_kv_heads, self.head_dim
                    )

                keys = self.k_norm(keys)
                values = self.v_norm(values)
                values = values.transpose(0, 2, 1, 3)

                keys = keys.transpose(0, 2, 1, 3)
                keys = self.rope(keys, offset=offset)

                if cache is not None:
                    keys, values = cache.update_and_fetch(keys, values)

            queries = queries.transpose(0, 2, 1, 3)
            queries = self.rope(queries, offset=offset)

            if mask is not None and isinstance(mask, mx.array):
                if mask.shape[-1] != keys.shape[-2]:
                    mask = mask[..., -keys.shape[-2] :]

            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(output), (keys, values), offset

        Attention.__call__ = _patched_attn

        # ── 3. Patch DecoderLayer.__call__ to propagate shared_kv/offset ──

        def _patched_layer(
            self, x, mask=None, cache=None, per_layer_input=None,
            shared_kv=None, offset=None,
        ):
            import mlx.nn as nn

            residual = x
            h = self.input_layernorm(x)
            h, shared_kv, offset = self.self_attn(
                h, mask, cache, shared_kv=shared_kv, offset=offset
            )
            h = self.post_attention_layernorm(h)
            h = residual + h

            residual = h
            if self.enable_moe:
                h1 = self.pre_feedforward_layernorm(h)
                h1 = self.mlp(h1)
                h1 = self.post_feedforward_layernorm_1(h1)
                top_k_indices, top_k_weights = self.router(h)
                h2 = self.pre_feedforward_layernorm_2(h)
                h2 = self.experts(h2, top_k_indices, top_k_weights)
                h2 = self.post_feedforward_layernorm_2(h2)
                h = h1 + h2
            else:
                h = self.pre_feedforward_layernorm(h)
                h = self.mlp(h)

            h = self.post_feedforward_layernorm(h)
            h = residual + h

            if (
                self.per_layer_input_gate is not None
                and self.per_layer_projection is not None
                and self.post_per_layer_input_norm is not None
                and per_layer_input is not None
            ):
                residual = h
                gate = self.per_layer_input_gate(h)
                gate = nn.gelu_approx(gate)
                gate = mx.multiply(gate, per_layer_input)
                gate = self.per_layer_projection(gate)
                gate = self.post_per_layer_input_norm(gate)
                h = residual + gate

            if self.layer_scalar is not None:
                h = h * self.layer_scalar

            return h, shared_kv, offset

        DecoderLayer.__call__ = _patched_layer

        # ── 4. Patch the model's layer loop for intermediates-based KV sharing ──

        from mlx_vlm.models.gemma4.language import Gemma4TextModel

        def _patched_model_call(
            self, inputs=None, inputs_embeds=None, mask=None, cache=None,
            per_layer_inputs=None, **kwargs,
        ):
            # Embed tokens (same as original Gemma4TextModel)
            if inputs_embeds is None:
                h = self.embed_tokens(inputs)
                h = h * self.embed_scale
            else:
                h = inputs_embeds

            # Per-layer input processing
            if self.hidden_size_per_layer_input:
                if inputs is not None and per_layer_inputs is None:
                    per_layer_inputs = self.get_per_layer_inputs(inputs)
                elif per_layer_inputs is not None:
                    target_len = h.shape[1]
                    if per_layer_inputs.shape[1] != target_len:
                        cache_offset = next(
                            (
                                int(c.offset) if not isinstance(c.offset, mx.array)
                                else int(c.offset.max().item())
                                for c in (cache or [])
                                if c is not None and hasattr(c, "offset")
                            ),
                            0,
                        )
                        max_start = max(per_layer_inputs.shape[1] - target_len, 0)
                        start = min(cache_offset, max_start)
                        per_layer_inputs = per_layer_inputs[:, start : start + target_len]
                if per_layer_inputs is not None or inputs is not None:
                    per_layer_inputs = self.project_per_layer_inputs(h, per_layer_inputs)

            # Build previous_kvs mapping if not cached
            if not hasattr(self, "_previous_kvs"):
                self._previous_kvs = list(range(len(self.layers)))
                num_shared = getattr(
                    self, "first_kv_shared_layer_idx", len(self.layers)
                )
                if num_shared < len(self.layers):
                    kvs_by_type = {}
                    for i in range(num_shared):
                        kvs_by_type[self.layers[i].layer_type] = i
                    for j in range(num_shared, len(self.layers)):
                        lt = self.layers[j].layer_type
                        if lt in kvs_by_type:
                            self._previous_kvs[j] = kvs_by_type[lt]

            if cache is None:
                cache = [None] * getattr(
                    self, "first_kv_shared_layer_idx", len(self.layers)
                )

            from mlx_lm.models.base import create_attention_mask

            if mask is None:
                full_idx = getattr(self, "first_full_cache_idx", 0)
                slide_idx = getattr(self, "first_sliding_cache_idx", 0)
                global_mask = create_attention_mask(
                    h,
                    cache[full_idx] if full_idx < len(cache) else None,
                )
                sliding_window_mask = create_attention_mask(
                    h,
                    cache[slide_idx] if slide_idx < len(cache) else None,
                    window_size=getattr(self, "window_size", None),
                )

            intermediates = [(None, None)] * len(self.layers)
            for i, layer in enumerate(self.layers):
                c = cache[self.layer_idx_to_cache_idx[i]]
                is_global = layer.layer_type == "full_attention"

                local_mask = mask
                if mask is None and is_global:
                    local_mask = global_mask
                elif mask is None:
                    local_mask = sliding_window_mask

                per_layer_input = None
                if per_layer_inputs is not None:
                    per_layer_input = per_layer_inputs[:, :, i, :]

                kvs, offset = intermediates[self._previous_kvs[i]]

                h, kvs, offset = layer(
                    h,
                    local_mask,
                    c,
                    per_layer_input=per_layer_input,
                    shared_kv=kvs,
                    offset=offset,
                )

                intermediates[i] = (kvs, offset)

            return self.norm(h)

        Gemma4TextModel.__call__ = _patched_model_call

        _gemma4_batched_decode_patched = True
        logger.debug("Applied Gemma 4 batched decode patch")
    except (ImportError, AttributeError) as e:
        logger.debug("Gemma 4 batched decode patch failed: %s", e)


# Models that only support a single image per request
SINGLE_IMAGE_ONLY_MODELS = {
    "llava_next",
    "llava-qwen2",
    "bunny-llama",
    "paligemma",
    "multi_modality",
    "mllama",
}

# Qwen-style VLMs: vision_tower takes (pixel_values, grid_thw)
_QWEN_VISION_MODELS = {
    "qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe",
    "qwen2_vl", "qwen2_5_vl",
}


class VLMBatchedEngine(BaseEngine):
    """
    VLM engine with continuous batching, tiered KV cache, and boundary snapshots.

    Extends the standard batched engine approach with vision-language model
    support. Uses VLMModelAdapter to inject pre-computed vision embeddings
    during prefill while maintaining full BatchGenerator compatibility.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        scheduler_config: Any | None = None,
        stream_interval: int = 1,
        enable_thinking: bool | None = None,
        model_settings: Any | None = None,
    ):
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._enable_thinking = enable_thinking
        self._model_settings = model_settings

        self._vlm_model = None
        self._processor = None
        self._tokenizer = None
        self._adapter = None
        self._engine = None
        self._loaded = False
        self._grammar_compiler = None
        self._grammar_compiler_init_attempted = False
        self._vision_cache = None
        self._vision_cache_enabled = True

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def model_type(self) -> str | None:
        if self._vlm_model is not None and hasattr(self._vlm_model, "config"):
            config = self._vlm_model.config
            if hasattr(config, "model_type"):
                return config.model_type
        return None

    @property
    def message_extractor(self):
        """Return the model-specific message extractor function, or ``None``."""
        try:
            from ..adapter.output_parser import detect_message_extractor
            model_config = {"model_type": self.model_type} if self.model_type else None
            return detect_message_extractor(self._model_name, model_config)
        except Exception:
            return None

    @property
    def is_ocr_model(self) -> bool:
        return (self.model_type or "") in OCR_MODEL_TYPES

    @property
    def grammar_compiler(self):
        """Lazily create and return a GrammarCompiler for this VLM model."""
        if self._grammar_compiler is not None:
            return self._grammar_compiler
        if self._grammar_compiler_init_attempted:
            return None
        self._grammar_compiler_init_attempted = True
        try:
            from ..api.grammar import create_grammar_compiler

            self._grammar_compiler = create_grammar_compiler(self._tokenizer, self._vlm_model)
            logger.info("GrammarCompiler initialized for %s", self._model_name)
        except Exception:
            from ..utils.install import get_install_method

            method = get_install_method()
            if method == "dmg":
                logger.info(
                    "Structured output is not available in the DMG version "
                    "(xgrammar requires torch which significantly increases app size). "
                    "Use the pip or Homebrew version for structured output support."
                )
            elif method == "homebrew":
                logger.info(
                    "Structured output requires xgrammar. "
                    "Reinstall with: brew reinstall omlx --with-grammar"
                )
            else:
                logger.info(
                    "Structured output requires xgrammar. "
                    "Install with: pip install 'omlx[grammar]'"
                )
        return self._grammar_compiler

    def _resolve_ocr_stop_token_ids(self) -> list[int]:
        """Convert OCR stop sequences to token IDs via the tokenizer.

        Caches the result after first call since the tokenizer doesn't change.
        """
        if hasattr(self, "_ocr_stop_ids_cache"):
            return self._ocr_stop_ids_cache

        ids: list[int] = []
        if self._tokenizer is None:
            return ids

        unk_id = getattr(self._tokenizer, "unk_token_id", None)
        for seq in OCR_EXTRA_STOP_SEQUENCES:
            try:
                token_id = self._tokenizer.convert_tokens_to_ids(seq)
                if token_id is not None and token_id != unk_id:
                    ids.append(token_id)
            except (AttributeError, KeyError, TypeError):
                pass

        self._ocr_stop_ids_cache = ids
        if ids:
            logger.debug(f"OCR stop token IDs resolved: {ids}")
        return ids

    async def start(self) -> None:
        """Load VLM model and processor via mlx-vlm, create engine with VLMModelAdapter."""
        if self._loaded:
            return

        from mlx_vlm.utils import load as vlm_load

        from ..engine_core import AsyncEngineCore, EngineConfig
        from ..scheduler import SchedulerConfig

        # Load VLM model on the global MLX executor to avoid blocking the event loop
        # while ensuring no concurrent Metal operations. See issue #85.
        from ..engine_core import get_mlx_executor

        def _load_vlm_sync():
            # Patch transformers bug: video_processor_class_from_name crashes
            # when torchvision is not available (extractors is None, `in` fails).
            # oMLX does not support video input, so we skip video processing.
            _patch_video_processor_bug()
            return vlm_load(self._model_name)

        loop = asyncio.get_running_loop()
        self._vlm_model, self._processor = await loop.run_in_executor(
            get_mlx_executor(), _load_vlm_sync
        )

        # Initialize vision feature cache
        vision_ssd_dir = None
        if self._scheduler_config and getattr(
            self._scheduler_config, "paged_ssd_cache_dir", None
        ):
            from pathlib import Path as _Path

            vision_ssd_dir = _Path(self._scheduler_config.paged_ssd_cache_dir) / "vision_features"
        self._vision_cache = VisionFeatureSSDCache(
            cache_dir=vision_ssd_dir,
            max_memory_entries=20,
        )
        logger.info("Vision feature cache enabled (SSD: %s)", vision_ssd_dir or "disabled")

        # Extract tokenizer from processor
        if hasattr(self._processor, "tokenizer"):
            self._tokenizer = self._processor.tokenizer
        else:
            self._tokenizer = self._processor

        # Build mlx-lm decode model for batched decode by sharing VLM weights.
        # mlx-vlm language models may produce degenerated output in batched
        # decode (e.g. gemma4 missing KV sharing between layers).
        # The LM model is constructed without evaluating initial random weights
        # (MLX lazy eval) then load_weights replaces them with VLM's arrays
        # by reference — zero additional GPU memory.
        self._lm_model = None
        try:
            from pathlib import Path as _Path

            from mlx.utils import tree_flatten
            from mlx_lm.utils import load_model

            def _build_decode_model():
                # Create LM model with lazy=True: reads disk headers for correct
                # quantized structure but does NOT evaluate weights → 0 GPU memory.
                lm_model, _ = load_model(
                    _Path(self._model_name), lazy=True
                )
                # Replace lazy weights with VLM's evaluated arrays by reference.
                # VLM params "model.*" map to LM "language_model.model.*".
                vlm_params = dict(tree_flatten(
                    self._vlm_model.language_model.parameters()
                ))
                lm_params = [
                    ("language_model." + k, v) for k, v in vlm_params.items()
                ]
                lm_model.load_weights(lm_params, strict=False)
                return lm_model

            self._lm_model = await loop.run_in_executor(
                get_mlx_executor(), _build_decode_model
            )
            logger.info("VLM decode model ready (weight sharing, zero-copy)")
        except Exception as e:
            logger.warning("mlx-lm decode model failed, using vlm fallback: %s", e)

        # Create VLM model adapter wrapping language_model
        self._adapter = VLMModelAdapter(
            self._vlm_model, decode_model=self._lm_model
        )

        # Create scheduler config
        scheduler_config = (
            copy.copy(self._scheduler_config) if self._scheduler_config
            else SchedulerConfig()
        )
        scheduler_config.model_name = self._model_name

        engine_config = EngineConfig(
            model_name=self._model_name,
            scheduler_config=scheduler_config,
            stream_interval=self._stream_interval,
        )

        # Create engine with adapter as the "model"
        # The adapter exposes .layers, .make_cache() for cache infrastructure
        self._engine = AsyncEngineCore(
            model=self._adapter,
            tokenizer=self._tokenizer,
            config=engine_config,
        )

        await self._engine.engine.start()

        # TurboQuant KV cache
        if self._model_settings is not None:
            tq_enabled = getattr(self._model_settings, "turboquant_kv_enabled", False)
            if tq_enabled:
                from ..patches.turboquant_attention import apply_turboquant_attention_patch
                apply_turboquant_attention_patch()
                tq_bits = float(getattr(self._model_settings, "turboquant_kv_bits", 4))
                self._engine.engine.scheduler._turboquant_kv_bits = tq_bits
                logger.info(f"TurboQuant KV cache enabled for VLM: {tq_bits} bits")

        # SpecPrefill: load draft model and pass to scheduler
        if self._model_settings is not None:
            specprefill_draft = getattr(self._model_settings, "specprefill_draft_model", None)
            specprefill_enabled = getattr(self._model_settings, "specprefill_enabled", False)
            if specprefill_enabled and specprefill_draft:
                try:
                    from mlx_lm import load as mlx_lm_load

                    def _load_draft():
                        draft_model, _ = mlx_lm_load(specprefill_draft)
                        return draft_model
                    draft_model = await loop.run_in_executor(get_mlx_executor(), _load_draft)
                    self._engine.engine.scheduler.set_specprefill_draft_model(
                        draft_model, draft_model_name=specprefill_draft
                    )
                    logger.info(f"SpecPrefill: draft model loaded ({specprefill_draft})")
                except Exception as e:
                    logger.error(f"SpecPrefill: draft model load failed: {e}")

        # Inject mlx-lm tool calling support into VLM tokenizer
        self._inject_tool_calling(self._tokenizer)

        self._loaded = True
        logger.info(f"VLMBatchedEngine loaded: {self._model_name}")

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._engine:
            await self._engine.stop()
            self._engine.engine.close()
        if self._vision_cache is not None:
            self._vision_cache.close()
            self._vision_cache = None
        self._engine = None
        self._vlm_model = None
        self._processor = None
        self._adapter = None
        self._tokenizer = None
        self._loaded = False
        logger.info("VLMBatchedEngine stopped")

    def _inject_tool_calling(self, tokenizer) -> None:
        """Inject tool calling attributes into VLM tokenizer.

        mlx-vlm's TokenizerWrapper lacks tool calling support (has_tool_calling,
        tool_parser, etc). We prefer mlx_vlm.tool_parsers which is a superset of
        mlx_lm's — it recognises additional markers such as Gemma4's <|tool_call>
        and loads the correct per-model parser.  Falls back to mlx_lm if the
        mlx_vlm.tool_parsers package is not present.
        """
        chat_template = getattr(tokenizer, "chat_template", None)
        if not chat_template:
            return

        # Prefer mlx_vlm.tool_parsers (superset; knows about Gemma4 etc.)
        try:
            from mlx_vlm.tool_parsers import (
                _infer_tool_parser,
                load_tool_module,
            )

            tool_parser_type = _infer_tool_parser(chat_template)
            if tool_parser_type is None:
                return
            try:
                tool_module = load_tool_module(tool_parser_type)
            except ImportError:
                logger.warning(f"VLM tool parser module not found: {tool_parser_type}")
                return
        except ImportError:
            # Fallback: mlx_lm only (no Gemma4 support)
            try:
                import importlib

                from mlx_lm.tokenizer_utils import (
                    _infer_tool_parser as _mlx_lm_infer,
                )
            except ImportError:
                return
            tool_parser_type = _mlx_lm_infer(chat_template)
            if tool_parser_type is None:
                return
            try:
                tool_module = importlib.import_module(
                    f"mlx_lm.tool_parsers.{tool_parser_type}"
                )
            except ImportError:
                logger.warning(f"VLM tool parser module not found: {tool_parser_type}")
                return

        tool_call_start = tool_module.tool_call_start
        tool_call_end = tool_module.tool_call_end

        # Validate tokens exist in vocab (same check as mlx-lm)
        vocab = tokenizer.get_vocab()
        if (tool_call_start and tool_call_start not in vocab) or (
            tool_call_end and tool_call_end not in vocab
        ):
            return

        # Set instance attributes on the mlx-vlm TokenizerWrapper.
        # Python's __getattr__ is only called when normal lookup fails,
        # so instance attributes take precedence over delegation to HF tokenizer.
        tokenizer.has_tool_calling = True
        tokenizer.tool_call_start = tool_call_start
        tokenizer.tool_call_end = tool_call_end
        tokenizer.tool_parser = tool_module.parse_tool_call

        logger.info(f"VLM tool calling enabled: parser={tool_parser_type}")

    @staticmethod
    def _count_content_parts(content: Any, part_types: set[str]) -> int:
        """Count multimodal parts in list content by type."""
        if not isinstance(content, list):
            return 0

        count = 0
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type", "")
            else:
                item_type = getattr(item, "type", "")
            if item_type in part_types:
                count += 1
        return count

    def _format_messages_for_vlm_template(
        self,
        messages: list[dict[str, Any]],
        num_images: int,
    ) -> list[dict[str, Any]]:
        """Format VLM messages with image tokens on image-bearing user turns."""
        from mlx_vlm.prompt_utils import extract_text_from_content, get_message_json

        model_type = self.model_type or getattr(self._vlm_model.config, "model_type", "")
        if not model_type:
            raise ValueError("Missing VLM model_type for chat template formatting")

        image_part_types = {"image", "image_url", "input_image"}
        has_explicit_images = any(
            isinstance(msg, dict)
            and self._count_content_parts(msg.get("content"), image_part_types) > 0
            for msg in messages
        )

        remaining_images = num_images
        assigned_fallback_images = False
        formatted_messages: list[dict[str, Any]] = []

        for msg in messages:
            if not isinstance(msg, dict):
                msg = {"role": "user", "content": str(msg)}

            role = msg.get("role", "user")
            raw_content = msg.get("content")
            content = extract_text_from_content(raw_content)

            msg_num_images = 0
            if role == "user":
                explicit_images = self._count_content_parts(raw_content, image_part_types)
                if explicit_images > 0 and remaining_images > 0:
                    msg_num_images = min(explicit_images, remaining_images)
                    remaining_images -= msg_num_images
                elif (
                    not has_explicit_images
                    and remaining_images > 0
                    and not assigned_fallback_images
                ):
                    msg_num_images = remaining_images
                    remaining_images = 0
                    assigned_fallback_images = True

            formatted_messages.append(
                get_message_json(
                    model_type,
                    content,
                    role,
                    skip_image_token=role != "user" or msg_num_images == 0,
                    skip_audio_token=True,
                    num_images=msg_num_images,
                    num_audios=0,
                )
            )

        return formatted_messages

    def _compute_vision_features(
        self, pixel_values: Any, extra_model_inputs: dict
    ) -> Optional[mx.array]:
        """Compute vision features for caching.

        Tries multiple strategies based on model architecture:
        1. model.encode_image() — upstream mlx-vlm API (e.g. gemma4)
        2. Direct vision_tower call for qwen-style models
        3. Direct vision_tower + projector for llava-style models
        4. Returns None for unsupported models

        Args:
            pixel_values: Preprocessed image tensors from prepare_inputs().
            extra_model_inputs: Additional model-specific inputs (e.g. image_grid_thw).

        Returns:
            Computed vision features (mx.array), or None if unsupported.
        """
        model = self._vlm_model
        model_type = self.model_type or ""

        # Strategy 1: upstream encode_image (gemma4 and future models)
        if hasattr(model, "encode_image"):
            return model.encode_image(pixel_values)

        # Strategy 2: qwen-style (vision_tower + grid_thw)
        if model_type in _QWEN_VISION_MODELS:
            grid_thw = extra_model_inputs.get("image_grid_thw")
            if grid_thw is None:
                grid_thw = extra_model_inputs.get("video_grid_thw")
            if grid_thw is None:
                return None
            dtype = model.vision_tower.patch_embed.proj.weight.dtype
            pv = mx.array(pixel_values) if not isinstance(pixel_values, mx.array) else pixel_values
            pv = pv.astype(dtype)
            result = model.vision_tower(pv, grid_thw)
            # qwen3_5 returns (hidden_states, _), qwen2_vl returns hidden_states
            if isinstance(result, tuple):
                return result[0]
            return result

        # Strategy 3: llava-style (vision_tower → layer select → projector)
        if model_type == "llava":
            pv = pixel_values
            if not isinstance(pv, mx.array):
                pv = mx.array(pv)
            *_, hidden_states = model.vision_tower(
                pv.transpose(0, 2, 3, 1), output_hidden_states=True
            )
            selected = hidden_states[model.vision_feature_layer]
            if isinstance(model.vision_feature_layer, int):
                if getattr(model, "vision_feature_select_strategy", "default") == "default":
                    selected = selected[:, 1:]
            else:
                hs_pool = [hidden_states[idx] for idx in model.vision_feature_layer]
                if getattr(model, "vision_feature_select_strategy", "default") == "default":
                    hs_pool = [hs[:, 1:] for hs in hs_pool]
                selected = mx.concatenate(hs_pool, axis=-1)
            return model.multi_modal_projector(selected)

        # Unsupported model: skip caching
        return None

    def _prepare_vision_inputs(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        chat_template_kwargs: dict[str, Any] | None = None,
        tools: list[dict] | None = None,
    ) -> Tuple[List[int], Optional[mx.array], Optional[Dict[str, Any]], Optional[str]]:
        """
        Run the full VLM preprocessing pipeline:
        1. Apply chat template with image placeholders
        2. Tokenize and preprocess images via processor
        3. Run vision encoder to produce merged embeddings
        4. Compute image hash for prefix cache

        Args:
            messages: Chat messages (text-only, images already extracted)
            images: List of PIL Image objects

        Returns:
            Tuple of (token_ids, inputs_embeds, extra_kwargs, image_hash):
            - token_ids: List of token IDs for BatchGenerator
            - inputs_embeds: Merged vision+text embeddings (or None if text-only)
            - extra_kwargs: Model-specific kwargs for language model
            - image_hash: SHA256 hash of images for prefix cache
        """
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs

        num_images = len(images)
        model_type = self.model_type or ""

        # Validate multi-image support
        if num_images > 1 and model_type in SINGLE_IMAGE_ONLY_MODELS:
            raise ValueError(
                f"Model {model_type} does not support multi-image chat. "
                f"Please use only 1 image."
            )

        # Apply VLM-specific chat template with image placeholders.
        # Build per-message placeholders in oMLX so image-bearing turns always
        # receive image tokens, regardless of conversation history shape.
        try:
            formatted_messages = self._format_messages_for_vlm_template(
                messages, num_images=num_images
            )
        except Exception as e:
            logger.debug(
                "Falling back to mlx-vlm apply_chat_template for VLM formatting: %s",
                e,
            )
            # Fallback to upstream formatter for unknown model/format edge cases.
            formatted_messages = apply_chat_template(
                self._processor,
                self._vlm_model.config,
                messages,
                num_images=num_images,
                return_messages=True,
            )

        # Strip partial field from messages (VLM always uses add_generation_prompt=True)
        detect_and_strip_partial(formatted_messages)
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self._enable_thinking is not None:
            template_kwargs["enable_thinking"] = self._enable_thinking
        # Per-model/request kwargs override global defaults (e.g. enable_thinking,
        # reasoning_effort).  This mirrors the text-only _apply_chat_template().
        if tools:
            template_kwargs["tools"] = tools
        if chat_template_kwargs:
            template_kwargs.update(chat_template_kwargs)

        # Use processor or its tokenizer for chat template application
        template_target = self._processor
        if not hasattr(template_target, "apply_chat_template"):
            template_target = getattr(self._processor, "tokenizer", self._processor)
        try:
            prompt = template_target.apply_chat_template(
                formatted_messages, **template_kwargs
            )
        except TypeError:
            # Fallback: template doesn't support some kwargs
            if chat_template_kwargs:
                for key in chat_template_kwargs:
                    template_kwargs.pop(key, None)
            template_kwargs.pop("enable_thinking", None)
            prompt = template_target.apply_chat_template(
                formatted_messages, **template_kwargs
            )
        except ValueError:
            # Processor has apply_chat_template but no chat_template set
            # (e.g. mlx-vlm custom processor without processor_config.json).
            # Fall back to processor.tokenizer which holds the actual template.
            fallback = getattr(self._processor, "tokenizer", None)
            if fallback is not None and fallback is not template_target:
                try:
                    prompt = fallback.apply_chat_template(
                        formatted_messages, **template_kwargs
                    )
                except TypeError:
                    if chat_template_kwargs:
                        for key in chat_template_kwargs:
                            template_kwargs.pop(key, None)
                    template_kwargs.pop("enable_thinking", None)
                    prompt = fallback.apply_chat_template(
                        formatted_messages, **template_kwargs
                    )
            else:
                raise

        # Tokenize text and preprocess images
        inputs = prepare_inputs(
            self._processor,
            images=images if images else None,
            prompts=[prompt] if isinstance(prompt, str) else prompt,
        )

        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        attention_mask = inputs.get("attention_mask")

        # Extract additional model-specific inputs (filter None values
        # since prepare_inputs may include them after mlx-vlm 348466f)
        extra_model_inputs = {
            k: v
            for k, v in inputs.items()
            if k not in ("input_ids", "attention_mask", "pixel_values")
            and v is not None
        }

        if pixel_values is not None and num_images > 0:
            # Compute image hash FIRST for vision feature cache lookup
            image_hash = compute_image_hash(images)

            # Build call kwargs from extra_model_inputs
            call_kwargs = dict(extra_model_inputs)

            # Try vision feature cache
            if self._vision_cache is not None and self._vision_cache_enabled and image_hash:
                cached_features = self._vision_cache.get(image_hash, self._model_name)
                if cached_features is not None:
                    call_kwargs["cached_image_features"] = cached_features
                    logger.debug("Vision feature cache hit: %s", image_hash[:16])
                else:
                    try:
                        features = self._compute_vision_features(
                            pixel_values, extra_model_inputs
                        )
                        if features is not None:
                            mx.eval(features)
                            self._vision_cache.put(
                                image_hash, self._model_name, features
                            )
                            call_kwargs["cached_image_features"] = features
                            logger.debug(
                                "Vision feature cache miss, stored: %s",
                                image_hash[:16],
                            )
                    except Exception:
                        logger.debug(
                            "Vision feature computation failed, using full pipeline",
                            exc_info=True,
                        )

            # Run vision encoder + embedding merge.
            # Pass attention_mask as 'mask' — mlx-vlm models (e.g. Gemma 3)
            # expect it as a positional/keyword arg named 'mask'.
            try:
                embed_features = self._vlm_model.get_input_embeddings(
                    input_ids, pixel_values, mask=attention_mask, **call_kwargs
                )
            except TypeError:
                # cached_image_features kwarg not supported — disable and retry
                if "cached_image_features" in call_kwargs:
                    logger.warning(
                        "cached_image_features not supported by %s, "
                        "disabling vision feature cache",
                        self.model_type,
                    )
                    self._vision_cache_enabled = False
                    call_kwargs.pop("cached_image_features")
                    embed_features = self._vlm_model.get_input_embeddings(
                        input_ids, pixel_values, mask=attention_mask, **call_kwargs
                    )
                else:
                    raise
            mx.eval(embed_features.inputs_embeds)

            # Convert InputEmbeddingsFeatures to dict for extra kwargs
            extra_kwargs = {}
            if hasattr(embed_features, "to_dict"):
                feat_dict = embed_features.to_dict()
                for k, v in feat_dict.items():
                    if k != "inputs_embeds" and v is not None:
                        extra_kwargs[k] = v

            # Extract token IDs as list
            token_ids = input_ids[0].tolist() if input_ids.ndim > 1 else input_ids.tolist()

            return token_ids, embed_features.inputs_embeds, extra_kwargs, image_hash
        else:
            # Text-only (no images in this message)
            token_ids = input_ids[0].tolist() if input_ids.ndim > 1 else input_ids.tolist()
            return token_ids, None, None, None

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Apply chat template for text-only messages (no images)."""
        if hasattr(self._tokenizer, "apply_chat_template"):
            # Strip partial field (VLM always uses add_generation_prompt=True)
            detect_and_strip_partial(messages)
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if tools:
                template_kwargs["tools"] = tools
            if self._enable_thinking is not None:
                template_kwargs["enable_thinking"] = self._enable_thinking
            if chat_template_kwargs:
                template_kwargs.update(chat_template_kwargs)

            try:
                return self._tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                if chat_template_kwargs:
                    for key in chat_template_kwargs:
                        template_kwargs.pop(key, None)
                template_kwargs.pop("tools", None)
                template_kwargs.pop("enable_thinking", None)
                return self._tokenizer.apply_chat_template(messages, **template_kwargs)
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return prompt + "\nassistant:"

    async def generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        vlm_inputs_embeds: Any = None,
        vlm_extra_kwargs: dict[str, Any] | None = None,
        vlm_image_hash: str | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """Generate a complete response (non-streaming)."""
        if not self._loaded:
            await self.start()

        # OCR models: add extra stop token IDs to prevent degeneration.
        # Sampling params (temperature, repetition_penalty, max_tokens) are
        # resolved by get_sampling_params() with OCR defaults as a fallback
        # layer, so admin/API overrides are respected.
        extra_stop_ids: list[int] = []
        if self.is_ocr_model:
            extra_stop_ids = self._resolve_ocr_stop_token_ids()

        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            xtc_probability=kwargs.get("xtc_probability", 0.0),
            xtc_threshold=kwargs.get("xtc_threshold", 0.1),
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stop=stop or [],
            stop_token_ids=extra_stop_ids or None,
            thinking_budget=kwargs.get("thinking_budget", None),
            compiled_grammar=kwargs.get("compiled_grammar", None),
            seed=kwargs.get("seed", None),
        )

        output = await self._engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            vlm_inputs_embeds=vlm_inputs_embeds,
            vlm_extra_kwargs=vlm_extra_kwargs,
            vlm_image_hash=vlm_image_hash,
        )

        text = clean_special_tokens(output.output_text)

        return GenerationOutput(
            text=text,
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            finish_reason=output.finish_reason,
            tool_calls=output.tool_calls,
            cached_tokens=output.cached_tokens,
        )

    async def stream_generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        vlm_inputs_embeds: Any = None,
        vlm_extra_kwargs: dict[str, Any] | None = None,
        vlm_image_hash: str | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream generation token by token."""
        if not self._loaded:
            await self.start()

        # OCR models: add extra stop token IDs to prevent degeneration.
        # Sampling params (temperature, repetition_penalty, max_tokens) are
        # resolved by get_sampling_params() with OCR defaults as a fallback
        # layer, so admin/API overrides are respected.
        extra_stop_ids: list[int] = []
        if self.is_ocr_model:
            extra_stop_ids = self._resolve_ocr_stop_token_ids()

        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            xtc_probability=kwargs.get("xtc_probability", 0.0),
            xtc_threshold=kwargs.get("xtc_threshold", 0.1),
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stop=stop or [],
            stop_token_ids=extra_stop_ids or None,
            thinking_budget=kwargs.get("thinking_budget", None),
            compiled_grammar=kwargs.get("compiled_grammar", None),
            seed=kwargs.get("seed", None),
        )

        # SpecPrefill: pass per-request overrides
        specprefill_kwargs = {}
        if kwargs.get("specprefill") is not None:
            specprefill_kwargs["specprefill"] = kwargs.pop("specprefill")
        if kwargs.get("specprefill_keep_pct") is not None:
            specprefill_kwargs["specprefill_keep_pct"] = kwargs.pop("specprefill_keep_pct")
        if kwargs.get("specprefill_threshold") is not None:
            specprefill_kwargs["specprefill_threshold"] = kwargs.pop("specprefill_threshold")
        if kwargs.get("specprefill_system_end") is not None:
            specprefill_kwargs["specprefill_system_end"] = kwargs.pop("specprefill_system_end")

        request_id = await self._engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            vlm_inputs_embeds=vlm_inputs_embeds,
            vlm_extra_kwargs=vlm_extra_kwargs,
            vlm_image_hash=vlm_image_hash,
            **specprefill_kwargs,
        )

        finished_normally = False
        try:
            async for output in self._engine.stream_outputs(request_id):
                text = clean_special_tokens(output.output_text)

                if output.finished:
                    finished_normally = True

                yield GenerationOutput(
                    text=text,
                    new_text=output.new_text,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finished=output.finished,
                    finish_reason=output.finish_reason,
                    tool_calls=output.tool_calls,
                    cached_tokens=output.cached_tokens,
                )
        except GeneratorExit:
            logger.info(f"[vlm_stream_generate] GeneratorExit for request {request_id}")
        finally:
            if not finished_normally:
                logger.info(f"[vlm_stream_generate] Aborting request {request_id}")
                await self._engine.abort_request(request_id)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """Chat completion with vision support (non-streaming)."""
        if not self._loaded:
            await self.start()

        loop = asyncio.get_running_loop()
        prompt, vlm_embeds, vlm_kwargs, image_hash = await loop.run_in_executor(
            self._engine._mlx_executor,
            self._process_chat_messages, messages, tools, kwargs,
        )

        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            vlm_inputs_embeds=vlm_embeds,
            vlm_extra_kwargs=vlm_kwargs,
            vlm_image_hash=image_hash,
            **kwargs,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream chat completion with vision support."""
        if not self._loaded:
            await self.start()

        # Run vision encoding on the MLX executor thread to avoid blocking
        # the event loop.  Blocking here (synchronous mx.eval) prevents
        # uvicorn from managing HTTP keep-alive connections, causing
        # TransferEncodingError on the next request (issue #80).
        loop = asyncio.get_running_loop()
        prompt, vlm_embeds, vlm_kwargs, image_hash = await loop.run_in_executor(
            self._engine._mlx_executor,
            self._process_chat_messages, messages, tools, kwargs,
        )

        # SpecPrefill: compute system prompt token count for protection.
        # Can't template system-only messages (most templates require user),
        # so compute by subtracting non-system from full prompt tokens.
        if kwargs.get("specprefill") is not False:
            non_system = [m for m in messages if m.get("role") not in ("system", "developer")]
            if len(non_system) < len(messages) and non_system:
                try:
                    non_system_prompt = self._tokenizer.apply_chat_template(
                        non_system, tokenize=False, add_generation_prompt=True,
                    )
                    full_tokens = len(self._tokenizer.encode(prompt))
                    non_system_tokens = len(self._tokenizer.encode(non_system_prompt))
                    system_end = full_tokens - non_system_tokens
                    if system_end > 0:
                        kwargs["specprefill_system_end"] = system_end
                except Exception as e:
                    logger.debug(f"SpecPrefill: system_end calc failed: {e}")

        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            vlm_inputs_embeds=vlm_embeds,
            vlm_extra_kwargs=vlm_kwargs,
            vlm_image_hash=image_hash,
            **kwargs,
        ):
            yield output

    def _apply_ocr_prompt(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply a default OCR prompt only when the user sends no text.

        OCR models (DeepSeek-OCR, GLM-OCR, DOTS-OCR) work best with specific
        prompt formats. When the user sends an image without any text, this
        injects the model's default OCR prompt. If the user provides their own
        text, it is preserved as-is so they can use custom prompts (e.g.
        structured extraction with JSON schema).

        Only activates when:
        - The model_type is in OCR_MODEL_PROMPTS
        - The last user message contains image content
        - The last user message has no meaningful text
        """
        model_type = self.model_type or ""
        if model_type not in OCR_MODEL_PROMPTS:
            return messages

        ocr_prompt = OCR_MODEL_PROMPTS[model_type]
        messages = copy.deepcopy(messages)

        # Find last user message
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                # Multi-part content: check if it has images
                has_image = any(
                    isinstance(p, dict) and p.get("type") == "image_url"
                    for p in content
                )
                if not has_image:
                    break
                # Check if user provided meaningful text
                user_text = " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ).strip()
                if user_text:
                    # User provided their own prompt, keep it
                    break
                # No user text — inject default OCR prompt
                new_content = [{"type": "text", "text": ocr_prompt}]
                new_content.extend(
                    p
                    for p in content
                    if not (isinstance(p, dict) and p.get("type") == "text")
                )
                msg["content"] = new_content
            break

        return messages

    def _process_chat_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None,
        kwargs: dict,
    ) -> Tuple[str | list[int], Any, dict | None, str | None]:
        """
        Process chat messages, extracting images and preparing VLM inputs.

        Returns:
            Tuple of (prompt_or_token_ids, vlm_embeds, vlm_kwargs, image_hash)
        """
        # Extract images from messages
        text_messages, images = extract_images_from_messages(messages)

        ct_kwargs = kwargs.pop("chat_template_kwargs", None)

        if images:
            # Apply OCR-specific prompt if applicable
            ocr_messages = self._apply_ocr_prompt(messages)

            # Convert tools for template format (same as text-only path)
            template_tools = convert_tools_for_template(tools) if tools else None

            # VLM path: prepare vision inputs
            token_ids, vlm_embeds, vlm_kwargs, image_hash = self._prepare_vision_inputs(
                ocr_messages, images,
                chat_template_kwargs=ct_kwargs,
                tools=template_tools,
            )
            return token_ids, vlm_embeds, vlm_kwargs, image_hash
        else:
            # Text-only path: standard chat template
            template_tools = convert_tools_for_template(tools) if tools else None
            prompt = self._apply_chat_template(
                text_messages, template_tools, chat_template_kwargs=ct_kwargs
            )
            return prompt, None, None, None

    def count_chat_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> int:
        """Count prompt tokens for chat messages (text-only approximation).

        For VLM messages with images, this counts only the text tokens.
        Image tokens are added during vision encoding and vary by model.
        """
        # Extract text-only version for token counting
        from ..utils.image import extract_images_from_messages
        text_messages, _ = extract_images_from_messages(messages)

        template_tools = convert_tools_for_template(tools) if tools else None
        prompt = self._apply_chat_template(
            text_messages, template_tools, chat_template_kwargs=chat_template_kwargs
        )
        return len(self._tokenizer.encode(prompt))

    def has_active_requests(self) -> bool:
        """Check if the engine has active in-flight requests."""
        engine_core = getattr(self, "_engine", None)
        if engine_core is not None:
            inner = getattr(engine_core, "engine", None)
            if inner is not None:
                collectors = getattr(inner, "_output_collectors", {})
                return len(collectors) > 0
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "engine_type": "vlm",
            "model_name": self._model_name,
            "loaded": self._loaded,
            "stream_interval": self._stream_interval,
        }
        if self._engine:
            stats.update(self._engine.get_stats())
        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics."""
        if self._engine:
            return self._engine.get_cache_stats()
        return None

    async def abort_all_requests(self) -> int:
        """Abort all active requests."""
        if self._engine and self._engine.engine:
            return await self._engine.engine.abort_all_requests()
        return 0
