# SPDX-License-Identifier: Apache-2.0
"""
DFlash engine for block diffusion speculative decoding.

This engine wraps dflash-mlx to provide 3-4x faster decoding on Apple Silicon.
For short/medium contexts it uses speculative decoding; for long contexts
(>DFLASH_MAX_CTX) it evicts dflash models and switches to omlx's BatchedEngine
or VLMBatchedEngine which have paged cache, SSD cache, and continuous batching.
"""

import asyncio
import copy
import gc
import logging
import os
import threading
from collections.abc import AsyncIterator
from typing import Any

import mlx.core as mx

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_special_tokens, detect_and_strip_partial
from .base import BaseEngine, GenerationOutput
from dflash_mlx.runtime import _ExactSmallProjPad, _target_text_model

logger = logging.getLogger(__name__)

DEFAULT_MAX_DFLASH_CTX = 4096

# Lock for toggling dflash hooks on the shared model.
_dflash_toggle_lock = threading.Lock()

# Reference count of active DFlash requests. Used to prevent disabling hooks
# while a DFlash generation is in-flight on the shared model.
_dflash_active_count = 0
_dflash_active_lock = threading.Lock()
_dflash_active_cond = threading.Condition(_dflash_active_lock)


def _get_text_model(model: Any) -> Any:
    """Get the inner text model from a wrapped target model."""
    return _target_text_model(model)

def _enable_dflash_hooks(model: Any) -> None:
    """Re-enable dflash-mlx hooks on all layers of the model.

    Sets _dflash_split_sdpa_enabled = True on self_attn layers so that the
    split_call wrapper runs its custom attention path instead of falling
    through to original_call. Also re-wraps _ExactSmallProjPad on linear_attn
    layers if they were unwrapped by a previous fallback request.

    Call this before any DFlash speculative decoding request.
    """
    text_model = _get_text_model(model)
    for layer in text_model.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn._dflash_split_sdpa_enabled = True
        if hasattr(layer, "linear_attn"):
            linear = layer.linear_attn
            for attr_name in ("in_proj_b", "in_proj_a"):
                proj = getattr(linear, attr_name, None)
                if proj is not None and type(proj).__name__ != "_ExactSmallProjPad":
                    # Was unwrapped by _disable_dflash_hooks — re-wrap it
                    setattr(linear, attr_name, _ExactSmallProjPad(proj))

    text_model._dflash_speculative_hooks_installed = True

def _disable_dflash_hooks(model: Any) -> None:
    """Disable dflash-mlx hooks on all layers of the model.

    Sets _dflash_split_sdpa_enabled = False on self_attn layers so that the
    split_call wrapper falls through to original_call. Also unwraps
    _ExactSmallProjPad on linear_attn layers.

    Call this before passing the model to BatchedEngine for the fallback path.
    """
    text_model = _get_text_model(model)
    for layer in text_model.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn._dflash_split_sdpa_enabled = False
        if hasattr(layer, "linear_attn"):
            linear = layer.linear_attn
            for attr_name in ("in_proj_b", "in_proj_a"):
                proj = getattr(linear, attr_name, None)
                if proj is not None and type(proj).__name__ == "_ExactSmallProjPad":
                    setattr(linear, attr_name, proj.linear)

    text_model._dflash_speculative_hooks_installed = False

def _enter_dflash_request() -> None:
    """
    Acquire a DFlash request slot. Blocks until no in-flight DFlash requests.
    Ensures there are not DFlash and non-DFlash requests being served at the same time.
    """
    global _dflash_active_count
    with _dflash_active_cond:
        while _dflash_active_count > 0:
            logger.debug("DFlash: waiting for in-flight requests to complete...")
            _dflash_active_cond.wait(timeout=30)
        _dflash_active_count += 1

def _exit_dflash_request() -> None:
    """Release a DFlash request slot and wake any waiting fallback engine."""
    global _dflash_active_count
    with _dflash_active_cond:
        _dflash_active_count -= 1
        _dflash_active_cond.notify_all()

class DFlashEngine(BaseEngine):
    """
    DFlash speculative decoding engine with per-request mode switching.

    For prompts within max_dflash_ctx tokens, uses block diffusion speculative
    decoding for 3-4x faster generation. For longer prompts, delegates to a fallback
    engine (BatchedEngine or VLMBatchedEngine) that provides paged cache, SSD cache,
    and continuous batching.

    The target model is loaded once and shared across both dflash and batched modes,
    avoiding model reload overhead when switching between them.
    """

    def __init__(
        self,
        model_name: str,
        draft_model_path: str,
        draft_quant_bits: int | None = None,
        model_settings: Any | None = None,
        fallback_engine_type: str = "batched",
        scheduler_config: Any | None = None,
    ):
        self._model_name = model_name
        self._draft_model_path = draft_model_path
        self._draft_quant_bits = draft_quant_bits
        self._model_settings = model_settings
        self._fallback_engine_type = fallback_engine_type
        self._scheduler_config = scheduler_config

        self._target_model = None
        self._draft_model = None
        self._tokenizer_obj = None
        self._executor_tokenizer = None
        self._loaded = False
        self._active_request = False
        self._model_type_str = None
        self._fallback_engine: BaseEngine | None = None
        if self._model_settings is not None and self._model_settings.dflash_max_ctx is not None:
            self._max_dflash_ctx = self._model_settings.dflash_max_ctx
        else:
            self._max_dflash_ctx = DEFAULT_MAX_DFLASH_CTX

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer_obj

    @property
    def model_type(self) -> str | None:
        return self._model_type_str

    async def start(self) -> None:
        if self._loaded:
            return

        from ..engine_core import get_mlx_executor

        loop = asyncio.get_running_loop()

        def _load_models():
            from dflash_mlx.runtime import load_target_bundle, load_draft_bundle

            model, tokenizer, meta = load_target_bundle(self._model_name)
            draft, draft_meta = load_draft_bundle(
                self._draft_model_path,
                quantize_draft=bool(self._draft_quant_bits),
            )
            return model, tokenizer, meta, draft

        result = await loop.run_in_executor(get_mlx_executor(), _load_models)
        self._target_model, self._tokenizer_obj, target_meta, self._draft_model = result

        from ..utils.model_loading import apply_post_load_transforms

        self._target_model = apply_post_load_transforms(
            self._target_model, self._model_settings
        )

        # Deep-copy tokenizer for executor-thread usage (dflash generation).
        # The original self._tokenizer_obj stays for event-loop operations
        # (encode, apply_chat_template, count_chat_tokens).
        # See: https://github.com/huggingface/tokenizers/issues/537
        self._executor_tokenizer = copy.deepcopy(self._tokenizer_obj)

        # Extract model_type from config
        config = target_meta.get("config", {})
        if isinstance(config, dict):
            self._model_type_str = config.get("model_type")
        elif hasattr(config, "model_type"):
            self._model_type_str = config.model_type

        self._loaded = True
        logger.info(
            f"DFlashEngine loaded: target={self._model_name}, "
            f"draft={self._draft_model_path}, "
            f"max_ctx={self._max_dflash_ctx}, "
            f"fallback={self._fallback_engine_type}"
        )

    async def _init_fallback_engine(self) -> None:
        """Create and start fallback engine sharing target model (no eviction).

        Before passing the model to BatchedEngine, disable dflash-mlx hooks
        that wrap self_attn/linear_attn.__call__ at the class level. These hooks
        add Python overhead and run a less-optimized attention path even during
        prefill, which can cause a ~25% throughput regression on the fallback path.

        Waits for any in-flight DFlash requests to complete before disabling hooks,
        ensuring no concurrent DFlash + batched execution on the shared model.
        """
        shared_model = self._target_model

        # Wait for all in-flight DFlash requests to finish before disabling hooks.
        _enter_dflash_request()  # ensures count is 0 when we proceed
        _exit_dflash_request()   # release our dummy slot

        with _dflash_toggle_lock:
            _disable_dflash_hooks(shared_model)

        shared_tokenizer = self._tokenizer_obj

        if self._fallback_engine_type == "vlm":
            from .vlm import VLMBatchedEngine
            self._fallback_engine = VLMBatchedEngine(
                model_name=self._model_name,
                scheduler_config=self._scheduler_config,
                model_settings=self._model_settings,
                model=shared_model,
                tokenizer=shared_tokenizer,
            )
        else:
            from .batched import BatchedEngine
            self._fallback_engine = BatchedEngine(
                model_name=self._model_name,
                scheduler_config=self._scheduler_config,
                model_settings=self._model_settings,
                model=shared_model,
                tokenizer=shared_tokenizer,
            )
        await self._fallback_engine.start()
        logger.info(
            f"DFlash fallback engine started: {self._fallback_engine_type}"
        )

    async def stop(self) -> None:
        if self._fallback_engine is not None:
            await self._fallback_engine.stop()
            self._fallback_engine = None
        self._target_model = None
        self._draft_model = None
        self._tokenizer_obj = None
        self._executor_tokenizer = None
        logger.info("DFlashEngine stopped")

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        if hasattr(self._tokenizer_obj, "apply_chat_template"):
            is_partial = detect_and_strip_partial(messages)
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": not is_partial,
            }
            if is_partial:
                template_kwargs["continue_final_message"] = True
            if tools:
                template_kwargs["tools"] = tools
            if chat_template_kwargs:
                template_kwargs.update(chat_template_kwargs)
            try:
                return self._tokenizer_obj.apply_chat_template(
                    messages, **template_kwargs
                )
            except TypeError:
                if chat_template_kwargs:
                    for key in chat_template_kwargs:
                        template_kwargs.pop(key, None)
                template_kwargs.pop("tools", None)
                template_kwargs.pop("enable_thinking", None)
                return self._tokenizer_obj.apply_chat_template(
                    messages, **template_kwargs
                )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return prompt + "\nassistant:"

    def count_chat_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> int:
        template_tools = convert_tools_for_template(tools) if tools else None
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=chat_template_kwargs
        )
        return len(self._tokenizer_obj.encode(prompt))

    def _run_generate_streaming(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        temperature: float,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Run dflash generation with streaming on MLX executor thread."""
        from dflash_mlx.generate import get_stop_token_ids
        from dflash_mlx.runtime import stream_dflash_generate

        try:
            stop_ids = get_stop_token_ids(self._executor_tokenizer)

            # Use streaming detokenizer for proper UTF-8 handling (CJK etc.)
            detokenizer = None
            try:
                from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer
                detokenizer = NaiveStreamingDetokenizer(self._executor_tokenizer)
            except ImportError:
                pass

            for event in stream_dflash_generate(
                target_model=self._target_model,
                tokenizer=self._executor_tokenizer,
                draft_model=self._draft_model,
                prompt="",
                max_new_tokens=max_tokens,
                stop_token_ids=stop_ids,
                prompt_tokens_override=prompt_tokens,
                temperature=temperature,
            ):
                event_type = event.get("event")

                if event_type == "token":
                    token_id = event["token_id"]
                    # Skip EOS/stop tokens from output
                    if token_id in stop_ids:
                        continue
                    if detokenizer is not None:
                        detokenizer.add_token(token_id)
                        text = detokenizer.last_segment
                    else:
                        text = self._executor_tokenizer.decode([token_id])
                    asyncio.run_coroutine_threadsafe(
                        queue.put((text, [token_id], False, None)), loop
                    )

                elif event_type == "summary":
                    gen_tokens = event.get("generation_tokens", 0)
                    accept_ratio = event.get("acceptance_ratio", 0)
                    cycles = event.get("cycles_completed", 0)
                    elapsed_us = event.get("elapsed_us", 0)
                    elapsed_s = elapsed_us / 1e6 if elapsed_us else 0
                    gen_tps = gen_tokens / elapsed_s if elapsed_s > 0 else 0
                    fallback = event.get("fallback_ar", False)
                    logger.info(
                        f"DFlash generation complete: "
                        f"{gen_tokens} tokens, "
                        f"{gen_tps:.1f} tok/s, "
                        f"acceptance={accept_ratio:.1%}, "
                        f"cycles={cycles}"
                        f"{', fallback=AR' if fallback else ''}"
                    )
                    metrics = {
                        "prompt_tokens": event.get("prompt_token_count", 0),
                        "completion_tokens": gen_tokens,
                        "acceptance_ratio": accept_ratio,
                        "cycles_completed": cycles,
                    }
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("", [], True, metrics)), loop
                    )

        except Exception as e:
            logger.error(f"DFlash streaming generation error: {e}")
            asyncio.run_coroutine_threadsafe(
                queue.put(("", [], True, {"error": str(e)})), loop
            )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()

        prompt_tokens = self._tokenizer_obj.encode(prompt)

        # Determine if we should use batched mode
        use_batched = len(prompt_tokens) >= self._max_dflash_ctx

        if use_batched:
            if self._fallback_engine is None:
                logger.info(
                    f"Switching to batched mode: prompt={len(prompt_tokens)} tokens "
                    f"(exceeds max_dflash_ctx={self._max_dflash_ctx})"
                )
                await self._init_fallback_engine()
            return await self._fallback_engine.generate(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty, stop=stop, **kwargs,
            )

        # Re-enable dflash hooks before speculative decoding path.
        # After a long-context fallback request, hooks may have been disabled.
        _enter_dflash_request()
        try:
            with _dflash_toggle_lock:
                _enable_dflash_hooks(self._target_model)

            from ..engine_core import get_mlx_executor
            from dflash_mlx.generate import get_stop_token_ids
            from dflash_mlx.runtime import generate_dflash_once

            loop = asyncio.get_running_loop()
            stop_ids = get_stop_token_ids(self._tokenizer_obj)

            def _run():
                return generate_dflash_once(
                    target_model=self._target_model,
                    tokenizer=self._executor_tokenizer,
                    draft_model=self._draft_model,
                    prompt="",
                    max_new_tokens=max_tokens,
                    stop_token_ids=stop_ids,
                    prompt_tokens_override=prompt_tokens,
                    temperature=temperature,
                )

            summary = await loop.run_in_executor(get_mlx_executor(), _run)

            generated = summary.get("generated_token_ids", [])
            text = self._tokenizer_obj.decode(generated, skip_special_tokens=True)
            text = clean_special_tokens(text)

            return GenerationOutput(
                text=text,
                tokens=generated,
                prompt_tokens=summary.get("prompt_token_count", len(prompt_tokens)),
                completion_tokens=summary.get("generation_tokens", len(generated)),
                finish_reason="stop",
            )
        finally:
            _exit_dflash_request()

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()

        prompt_tokens = self._tokenizer_obj.encode(prompt)

        # Determine if we should use batched mode
        use_batched = len(prompt_tokens) >= self._max_dflash_ctx

        if use_batched:
            if self._fallback_engine is None:
                logger.info(
                    f"Switching to batched mode: prompt={len(prompt_tokens)} tokens "
                    f"(exceeds max_dflash_ctx={self._max_dflash_ctx})"
                )
                await self._init_fallback_engine()
            async for output in self._fallback_engine.stream_generate(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty, stop=stop, **kwargs,
            ):
                yield output
            return

        # Re-enable dflash hooks before speculative decoding path.
        _enter_dflash_request()
        try:
            with _dflash_toggle_lock:
                _enable_dflash_hooks(self._target_model)

            prompt_len = len(prompt_tokens)
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            from ..engine_core import get_mlx_executor
            loop.run_in_executor(
                get_mlx_executor(),
                self._run_generate_streaming,
                prompt_tokens,
                max_tokens,
                temperature,
                queue,
                loop,
            )

            total_text = ""
            total_completion = 0

            while True:
                new_text, new_tokens, finished, metrics = await queue.get()

                total_text += new_text
                total_completion += len(new_tokens)

                finish_reason = None
                if finished:
                    finish_reason = "stop"
                    if metrics and metrics.get("error"):
                        finish_reason = "error"

                yield GenerationOutput(
                    text=total_text,
                    new_text=new_text,
                    tokens=new_tokens,
                    prompt_tokens=prompt_len,
                    completion_tokens=total_completion,
                    finished=finished,
                    finish_reason=finish_reason,
                )

                if finished:
                    break
        finally:
            _exit_dflash_request()

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
        if not self._loaded:
            await self.start()

        template_tools = convert_tools_for_template(tools) if tools else None
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=ct_kwargs
        )

        return await self.generate(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty, **kwargs,
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
        if not self._loaded:
            await self.start()

        template_tools = convert_tools_for_template(tools) if tools else None
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=ct_kwargs
        )

        async for output in self.stream_generate(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty, **kwargs,
        ):
            yield output

    def has_active_requests(self) -> bool:
        if self._fallback_engine is not None and self._fallback_engine.has_active_requests():
            return True
        return self._active_request

    def get_stats(self) -> dict[str, Any]:
        return {
            "engine_type": "dflash",
            "model_name": self._model_name,
            "draft_model": self._draft_model_path,
            "max_dflash_ctx": self._max_dflash_ctx,
            "fallback_engine_type": self._fallback_engine_type,
            "loaded": self._loaded,
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        # The DFlashEngine doesn't use paged KV cache / SSD tiering, so this is a stub.
        # The "fallback" BatchedEngine reports it's own cache stats separately.
        return None
