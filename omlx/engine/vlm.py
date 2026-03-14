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
from ..api.utils import clean_special_tokens
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

_video_processor_patched = False


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


# Models that only support a single image per request
SINGLE_IMAGE_ONLY_MODELS = {
    "llava_next",
    "llava-qwen2",
    "bunny-llama",
    "paligemma",
    "multi_modality",
    "mllama",
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
    ):
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._enable_thinking = enable_thinking

        self._vlm_model = None
        self._processor = None
        self._tokenizer = None
        self._adapter = None
        self._engine = None
        self._loaded = False

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
    def is_ocr_model(self) -> bool:
        return (self.model_type or "") in OCR_MODEL_TYPES

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

        # Extract tokenizer from processor
        if hasattr(self._processor, "tokenizer"):
            self._tokenizer = self._processor.tokenizer
        else:
            self._tokenizer = self._processor

        # Create VLM model adapter wrapping language_model
        self._adapter = VLMModelAdapter(self._vlm_model)

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

        # Inject mlx-lm tool calling support into VLM tokenizer
        self._inject_tool_calling(self._tokenizer)

        self._loaded = True
        logger.info(f"VLMBatchedEngine loaded: {self._model_name}")

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._engine:
            await self._engine.stop()
            self._engine.engine.close()
        self._engine = None
        self._vlm_model = None
        self._processor = None
        self._adapter = None
        self._tokenizer = None
        self._loaded = False
        logger.info("VLMBatchedEngine stopped")

    def _inject_tool_calling(self, tokenizer) -> None:
        """Inject mlx-lm's tool calling attributes into VLM tokenizer.

        mlx-vlm's TokenizerWrapper lacks tool calling support (has_tool_calling,
        tool_parser, etc). We reuse mlx-lm's _infer_tool_parser() to detect the
        parser type from the chat template, then set the attributes directly on
        the wrapper instance so parse_tool_calls() can use native tool parsing.
        """
        try:
            from mlx_lm.tokenizer_utils import _infer_tool_parser
        except ImportError:
            return

        chat_template = getattr(tokenizer, "chat_template", None)
        if not chat_template:
            return

        tool_parser_type = _infer_tool_parser(chat_template)
        if tool_parser_type is None:
            return

        try:
            import importlib

            tool_module = importlib.import_module(
                f"mlx_lm.tool_parsers.{tool_parser_type}"
            )
        except ImportError:
            logger.warning(
                f"VLM tool parser module not found: {tool_parser_type}"
            )
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
            # Run vision encoder + embedding merge.
            # Pass attention_mask as 'mask' — mlx-vlm models (e.g. Gemma 3)
            # expect it as a positional/keyword arg named 'mask'.
            embed_features = self._vlm_model.get_input_embeddings(
                input_ids, pixel_values, mask=attention_mask, **extra_model_inputs
            )
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

            # Compute image hash for prefix cache
            image_hash = compute_image_hash(images)

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

        # OCR models: force temperature=0.0 and add extra stop token IDs
        # to prevent degeneration (repeated <|user|>, <|im_start|>, etc.).
        extra_stop_ids: list[int] = []
        if self.is_ocr_model:
            temperature = 0.0
            extra_stop_ids = self._resolve_ocr_stop_token_ids()

        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stop=stop or [],
            stop_token_ids=extra_stop_ids or None,
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

        # OCR models: force temperature=0.0 and add extra stop token IDs.
        extra_stop_ids: list[int] = []
        if self.is_ocr_model:
            temperature = 0.0
            extra_stop_ids = self._resolve_ocr_stop_token_ids()

        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stop=stop or [],
            stop_token_ids=extra_stop_ids or None,
        )

        request_id = await self._engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            vlm_inputs_embeds=vlm_inputs_embeds,
            vlm_extra_kwargs=vlm_extra_kwargs,
            vlm_image_hash=vlm_image_hash,
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
