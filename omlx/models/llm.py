# SPDX-License-Identifier: Apache-2.0
"""
MLX Language Model wrapper.

This module provides a wrapper around mlx-lm for LLM inference,
integrating with vLLM's model execution system.
"""

import logging
from dataclasses import dataclass
from typing import Iterator

from ..api.utils import detect_and_strip_partial
from ..utils.tokenizer import get_tokenizer_config

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Output from text generation."""

    text: str
    tokens: list[int]
    finish_reason: str | None = None


@dataclass
class StreamingOutput:
    """Streaming output chunk."""

    text: str
    token: int
    finished: bool = False
    finish_reason: str | None = None


class MLXLanguageModel:
    """
    Wrapper around mlx-lm for LLM inference.

    This class provides a unified interface for loading and running
    inference on language models using Apple's MLX framework.

    Example:
        >>> model = MLXLanguageModel("mlx-community/Llama-3.2-3B-Instruct-4bit")
        >>> output = model.generate("Hello, how are you?", max_tokens=100)
        >>> print(output.text)
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_name: str | None = None,
        trust_remote_code: bool = False,
    ):
        """
        Initialize the MLX language model.

        Args:
            model_name: HuggingFace model name or local path
            tokenizer_name: Optional separate tokenizer name
            trust_remote_code: Whether to trust remote code
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.trust_remote_code = trust_remote_code

        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            return

        try:
            from mlx_lm import load

            from ..utils.model_loading import (
                maybe_apply_pre_load_patches,
                maybe_load_custom_quantization,
            )

            logger.info(f"Loading model: {self.model_name}")

            # Build tokenizer config with model-specific fixes
            tokenizer_config = get_tokenizer_config(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )

            # Apply pre-load patches for models that need module injection
            # before mlx_lm.load runs (e.g. DeepSeek V4 PR 1192).
            maybe_apply_pre_load_patches(self.model_name)

            custom_loaded = maybe_load_custom_quantization(
                self.model_name,
                tokenizer_config=tokenizer_config,
                trust_remote_code=self.trust_remote_code,
                is_vlm=False,
            )
            if custom_loaded is not None:
                model, processor = custom_loaded
                self.model = model
                self.tokenizer = getattr(processor, "tokenizer", processor)
            else:
                self.model, self.tokenizer = load(
                    self.model_name,
                    tokenizer_config=tokenizer_config,
                )

            self._loaded = True
            logger.info(f"Model loaded successfully: {self.model_name}")

        except ImportError:
            raise ImportError(
                "mlx-lm is required for LLM inference. "
                "Install with: pip install mlx-lm"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _create_sampler(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """Create a sampler for text generation."""
        from ..utils.sampling import make_sampler

        return make_sampler(
            temp=temperature,
            top_p=top_p,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
    ) -> GenerationOutput:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens
            stop: List of stop sequences

        Returns:
            GenerationOutput with generated text and tokens
        """
        if not self._loaded:
            self.load()

        from mlx_lm import generate

        # Create sampler with parameters
        sampler = self._create_sampler(temperature, top_p)

        # Generate text
        output_text = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )

        # Tokenize output to get token IDs
        tokens = self.tokenizer.encode(output_text)

        # Determine finish reason
        finish_reason = "length" if len(tokens) >= max_tokens else "stop"

        return GenerationOutput(
            text=output_text,
            tokens=tokens,
            finish_reason=finish_reason,
        )

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
    ) -> Iterator[StreamingOutput]:
        """
        Stream text generation token by token.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens
            stop: List of stop sequences

        Yields:
            StreamingOutput for each generated token
        """
        if not self._loaded:
            self.load()

        from mlx_lm import stream_generate

        # Create sampler with parameters
        sampler = self._create_sampler(temperature, top_p)

        token_count = 0
        accumulated_text = ""

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            token_count += 1
            # response.text is the new token text (not accumulated)
            new_text = response.text
            accumulated_text += new_text

            # Check for stop sequences
            should_stop = False
            if stop:
                for stop_seq in stop:
                    if stop_seq in accumulated_text:
                        should_stop = True
                        break

            finished = should_stop or token_count >= max_tokens
            finish_reason = None
            if finished:
                finish_reason = "stop" if should_stop else "length"

            yield StreamingOutput(
                text=new_text,
                token=response.token if hasattr(response, "token") else 0,
                finished=finished,
                finish_reason=finish_reason,
            )

            if finished:
                break

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list | None = None,
        enable_thinking: bool | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate a chat response.

        Args:
            messages: List of chat messages [{"role": "user", "content": "..."}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            tools: Optional list of tools for function calling
            enable_thinking: Enable thinking mode for reasoning models (passed to chat_template_kwargs)
            **kwargs: Additional generation parameters

        Returns:
            GenerationOutput with the assistant's response
        """
        if not self._loaded:
            self.load()

        # Apply chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            is_partial = detect_and_strip_partial(messages)
            # Build kwargs for apply_chat_template
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": not is_partial,
            }
            if is_partial:
                template_kwargs["continue_final_message"] = True

            # Add tools if provided and supported
            if tools:
                template_kwargs["tools"] = tools

            # Add enable_thinking if specified
            if enable_thinking is not None:
                template_kwargs["enable_thinking"] = enable_thinking

            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    **template_kwargs,
                )
            except TypeError:
                # Tokenizer doesn't support some parameter, try without tools and enable_thinking
                template_kwargs.pop("tools", None)
                template_kwargs.pop("enable_thinking", None)
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    **template_kwargs,
                )
        else:
            # Fallback: simple concatenation
            prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
            prompt += "\nassistant:"

        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._loaded:
            return {"loaded": False, "model_name": self.model_name}

        info = {
            "loaded": True,
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
        }

        # Try to get model config
        if hasattr(self.model, "config"):
            config = self.model.config
            info.update(
                {
                    "vocab_size": getattr(config, "vocab_size", None),
                    "hidden_size": getattr(config, "hidden_size", None),
                    "num_layers": getattr(config, "num_hidden_layers", None),
                    "num_heads": getattr(config, "num_attention_heads", None),
                }
            )

        return info

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"<MLXLanguageModel model={self.model_name} status={status}>"
