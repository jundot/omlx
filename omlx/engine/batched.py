# SPDX-License-Identifier: Apache-2.0
"""
Batched engine for continuous batching with multiple concurrent users.

This engine wraps AsyncEngineCore to provide continuous batching
for better throughput when serving multiple concurrent requests.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_output_text
from ..utils.tokenizer import get_tokenizer_config
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)

# Optional Harmony adapter import
try:
    from ..adapter.harmony import preprocess_harmony_messages

    HAS_HARMONY_ADAPTER = True
except ImportError:
    HAS_HARMONY_ADAPTER = False
    preprocess_harmony_messages = None  # type: ignore


class BatchedEngine(BaseEngine):
    """
    Batched engine for continuous batching.

    This engine provides better throughput when serving multiple
    concurrent users by batching requests together.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        scheduler_config: Any | None = None,
        stream_interval: int = 1,
        enable_thinking: bool | None = None,
    ):
        """
        Initialize the batched engine.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Whether to trust remote code
            scheduler_config: Optional scheduler configuration
            stream_interval: Tokens to batch before streaming (1=every token)
            enable_thinking: Enable thinking mode for reasoning models (passed to chat_template_kwargs)
        """
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._enable_thinking = enable_thinking

        self._model = None
        self._tokenizer = None
        self._engine = None
        self._loaded = False

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        return self._tokenizer

    @property
    def model_type(self) -> str | None:
        """Get the model type from config (e.g., 'gpt_oss', 'llama', 'qwen2')."""
        if self._model is None:
            return None
        # Try different ways to access model_type
        try:
            if hasattr(self._model, 'config'):
                config = self._model.config
                if hasattr(config, 'model_type'):
                    model_type = config.model_type
                    return model_type if isinstance(model_type, str) else None
                elif isinstance(config, dict):
                    model_type = config.get('model_type')
                    return model_type if isinstance(model_type, str) else None
            if hasattr(self._model, 'args'):
                args = self._model.args
                if hasattr(args, 'model_type'):
                    model_type = args.model_type
                    return model_type if isinstance(model_type, str) else None
        except Exception as e:
            logger.debug(f"Error getting model_type: {e}")
        return None

    def _preprocess_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Preprocess messages for model-specific formats.

        Currently handles Harmony (gpt-oss) models.

        Args:
            messages: List of chat messages

        Returns:
            Preprocessed messages
        """
        if self.model_type == "gpt_oss" and HAS_HARMONY_ADAPTER:
            return preprocess_harmony_messages(messages)
        return messages

    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        if self._loaded:
            return

        import asyncio

        from mlx_lm import load

        from ..engine_core import AsyncEngineCore, EngineConfig
        from ..scheduler import SchedulerConfig

        # Build tokenizer config with model-specific fixes
        tokenizer_config = get_tokenizer_config(
            self._model_name,
            trust_remote_code=self._trust_remote_code,
        )

        # Load model and tokenizer in a separate thread to avoid blocking the event loop.
        # This allows /admin and /health endpoints to remain responsive during model loading.
        def _load_model_sync():
            return load(
                self._model_name,
                tokenizer_config=tokenizer_config,
            )

        self._model, self._tokenizer = await asyncio.to_thread(_load_model_sync)

        # Create engine config
        scheduler_config = self._scheduler_config or SchedulerConfig()
        scheduler_config.model_name = self._model_name  # Ensure cache isolation per model
        engine_config = EngineConfig(
            model_name=self._model_name,
            scheduler_config=scheduler_config,
            stream_interval=self._stream_interval,
        )

        # Create async engine
        self._engine = AsyncEngineCore(
            model=self._model,
            tokenizer=self._tokenizer,
            config=engine_config,
        )

        await self._engine.engine.start()
        self._loaded = True
        logger.info(f"BatchedEngine loaded: {self._model_name}")

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._engine:
            await self._engine.stop()
            self._engine.engine.close()
        self._engine = None
        self._model = None
        self._tokenizer = None
        self._loaded = False
        logger.info("BatchedEngine stopped")

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ) -> str:
        """Apply chat template to messages."""
        if hasattr(self._tokenizer, 'apply_chat_template'):
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if tools:
                template_kwargs["tools"] = tools
            if self._enable_thinking is not None:
                template_kwargs["enable_thinking"] = self._enable_thinking

            try:
                return self._tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                # Tokenizer doesn't support some parameter, try without tools and enable_thinking
                template_kwargs.pop("tools", None)
                template_kwargs.pop("enable_thinking", None)
                return self._tokenizer.apply_chat_template(messages, **template_kwargs)
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return prompt + "\nassistant:"

    def count_chat_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ) -> int:
        """
        Count prompt tokens for chat messages after applying chat template.

        Args:
            messages: List of chat messages
            tools: Optional tool definitions

        Returns:
            Number of prompt tokens
        """
        messages = self._preprocess_messages(messages)
        template_tools = convert_tools_for_template(tools) if tools else None
        prompt = self._apply_chat_template(messages, template_tools)
        return len(self._tokenizer.encode(prompt))

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate a complete response (non-streaming).

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty (1.0 = disabled)
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with complete text
        """
        if not self._loaded:
            await self.start()

        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=stop or [],
        )

        output = await self._engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
        )

        text = clean_output_text(output.output_text)

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
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream generation token by token.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty (1.0 = disabled)
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        if not self._loaded:
            await self.start()

        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=stop or [],
        )

        request_id = await self._engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
        )

        finished_normally = False
        try:
            async for output in self._engine.stream_outputs(request_id):
                text = clean_output_text(output.output_text)

                # Set finished_normally BEFORE yield, because the consumer
                # may stop iterating after receiving the final output,
                # which triggers GeneratorExit at the yield point -
                # code after yield would never execute.
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
            # Client disconnected
            logger.info(f"[stream_generate] GeneratorExit caught for request {request_id}")
        finally:
            # Abort the request if client disconnected before completion
            if not finished_normally:
                logger.info(f"[stream_generate] Aborting request {request_id} (finished_normally={finished_normally})")
                await self._engine.abort_request(request_id)
            else:
                logger.debug(f"[stream_generate] Request {request_id} finished normally")

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Chat completion (non-streaming).

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty (1.0 = disabled)
            tools: Optional tool definitions
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with assistant response
        """
        if not self._loaded:
            await self.start()

        # Preprocess messages for Harmony (gpt-oss) models
        messages = self._preprocess_messages(messages)

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Apply chat template
        prompt = self._apply_chat_template(messages, template_tools)

        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream chat completion token by token.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty (1.0 = disabled)
            tools: Optional tool definitions
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        if not self._loaded:
            await self.start()

        # Preprocess messages for Harmony (gpt-oss) models
        messages = self._preprocess_messages(messages)

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Apply chat template
        prompt = self._apply_chat_template(messages, template_tools)

        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            **kwargs,
        ):
            yield output

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "engine_type": "batched",
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
